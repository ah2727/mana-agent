"""
mana_analyzer.llm.coding_agent

Coding agent wrapper with:
- mutation tools (write_file/apply_patch)
- structured flow/checklist planning
- anti-loop tool policy (search/read budgets, duplicate search guards)
- flow-memory continuity integration
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, Optional, Protocol, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

from mana_analyzer.llm.prompts import (
    CODING_AGENT_RECOGNITION_PROMPT,
    CODING_FLOW_MEMORY_PROMPT,
    CODING_FLOW_PLANNER_PROMPT,
)
from mana_analyzer.services.coding_memory_service import CodingMemoryService
from mana_analyzer.tools import build_apply_patch_tool, build_write_file_tool

logger = logging.getLogger(__name__)

CODING_SYSTEM_PROMPT = """\
You are a coding agent operating inside a repository.

Rules:
- Start with a short execution plan (files to inspect, intended edits, verification).
- Prefer repository-local tools first (semantic_search/read_file/run_command) before internet lookup.
- Keep internet search calls minimal; do not repeat near-identical search queries.
- Prefer apply_patch (unified diff) for edits to existing files.
- Use write_file only for new files or when explicitly asked to overwrite.
- Only modify files under src/ and tests/ unless the user explicitly asks otherwise.
- After changes, aim for clean static checks; avoid unused imports and obvious style issues.
- When you create new public functions/classes, add docstrings and type hints.

PATCH TOOL STRICT FORMAT:

When you call apply_patch, the patch MUST be a git-unified diff that `git apply` accepts.

REQUIRED structure:
- Starts with: diff --git a/<path> b/<path>
- Includes: --- a/<path> and +++ b/<path>
- Includes at least one @@ hunk with context lines.
- Paths must be repo-relative (no /absolute, no C:\\, no ..)

FORBIDDEN:
- Do NOT output "*** Begin Patch", "*** Update File", "*** End Patch" (not valid for git apply).
- Do NOT wrap the diff in ``` fences unless asked.

Workflow:
1) First call apply_patch(check_only=true) with the unified diff.
2) If ok=true, call apply_patch(check_only=false) with the same diff.
3) If check fails twice OR patch attempts produce no repo changes, STOP patching and switch to write_file fallback:
   - read_file the target
   - generate full updated file content
   - write_file in chunks with part_index, then finalize=true

"""


class FlowStep(BaseModel):
    id: str
    title: str
    reason: str
    status: Literal["pending", "in_progress", "done", "blocked"] = "pending"
    requires_tools: list[str] = Field(default_factory=list)


class FlowChecklist(BaseModel):
    objective: str
    constraints: list[str] = Field(default_factory=list)
    acceptance: list[str] = Field(default_factory=list)
    steps: list[FlowStep] = Field(default_factory=list)
    next_action: str = ""


class ExecutionDecision(BaseModel):
    phase: Literal["discover", "inspect", "edit", "verify", "answer", "blocked"]
    tool_call_allowed: bool
    why: str


class AskAgentLike(Protocol):
    tools: list[Any]

    def ask(self, question: str, **kwargs: Any) -> Any:  # pragma: no cover
        ...


def _as_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_as_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _as_jsonable(v) for k, v in obj.items()}
    return obj


class CodingAgent:
    _EDIT_INTENT_TOKENS = (
        "fix",
        "bug",
        "issue",
        "patch",
        "edit",
        "update",
        "modify",
        "change",
        "implement",
        "add",
        "remove",
        "delete",
        "create",
        "write",
        "refactor",
        "rename",
        "cleanup",
    )
    _WEB_INTENT_TOKENS = ("latest", "internet", "online", "web", "news", "search web")
    _EXPLICIT_FILE_RE = re.compile(r"(?i)\b([A-Za-z0-9_\-./]+\.[A-Za-z0-9_]+)\b")

    # Fallback policy: if apply_patch is attempted >= this many times and no changes appear, force write_file fallback.
    _PATCH_FAILURE_FALLBACK_THRESHOLD = 2

    def __init__(
        self,
        *,
        api_key: str,
        repo_root: Path,
        ask_agent: AskAgentLike,
        base_url: str | None = None,
        allowed_prefixes: Optional[Sequence[str]] = ("src/", "tests/"),
        system_prompt: str = CODING_SYSTEM_PROMPT,
        coding_memory_service: CodingMemoryService | None = None,
        coding_memory_enabled: bool = True,
        plan_max_steps: int = 8,
        search_budget: int = 4,
        read_budget: int = 6,
        require_read_files: int = 2,
        repo_only_internet_default: bool = True,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.repo_root = repo_root.resolve()
        self.ask_agent: AskAgentLike = ask_agent
        self.allowed_prefixes = allowed_prefixes
        self.system_prompt = system_prompt
        self.coding_memory_service = coding_memory_service
        self.coding_memory_enabled = bool(coding_memory_enabled)
        self._current_flow_id: str | None = None

        self.plan_max_steps = max(1, int(plan_max_steps))
        self.search_budget = max(1, int(search_budget))
        self.read_budget = max(1, int(read_budget))
        self.require_read_files = max(1, int(require_read_files))
        self.repo_only_internet_default = bool(repo_only_internet_default)

        planner_kwargs = {
            "api_key": api_key,
            "model": str(getattr(ask_agent, "model", "gpt-4.1-mini")),
        }
        if base_url:
            planner_kwargs["base_url"] = base_url
        self.planner_llm = ChatOpenAI(**planner_kwargs)

        self.ask_agent.tools.extend(
            [
                build_write_file_tool(repo_root=self.repo_root, allowed_prefixes=self.allowed_prefixes),
                build_apply_patch_tool(repo_root=self.repo_root, allowed_prefixes=self.allowed_prefixes),
            ]
        )

    def _looks_like_edit_request(self, request: str) -> bool:
        lowered = request.lower()
        return any(token in lowered for token in self._EDIT_INTENT_TOKENS)

    def _allows_web_search(self, request: str) -> bool:
        lowered = request.lower()
        return any(token in lowered for token in self._WEB_INTENT_TOKENS)

    def _effective_system_prompt_for(self, request: str, *, flow_context: str | None = None) -> str:
        prompt = self.system_prompt
        if self._looks_like_edit_request(request):
            prompt = f"{prompt}\n\n{CODING_AGENT_RECOGNITION_PROMPT}"
        if flow_context:
            prompt = (
                f"{prompt}\n\n{CODING_FLOW_MEMORY_PROMPT}\n\n"
                f"Flow context:\n{flow_context.strip()}"
            )
        return prompt

    def _plan_checklist(self, request: str, *, flow_context: str | None = None) -> tuple[FlowChecklist | None, list[str]]:
        warnings: list[str] = []
        user_prompt = (
            f"User request:\n{request}\n\n"
            f"Max steps: {self.plan_max_steps}\n\n"
            f"Flow context:\n{(flow_context or 'none').strip()}\n"
        )
        messages = [
            SystemMessage(content=CODING_FLOW_PLANNER_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        def _parse(text: str) -> FlowChecklist:
            payload = json.loads(text)
            return FlowChecklist.model_validate(payload)

        try:
            first = self.planner_llm.invoke(messages)
            raw = str(getattr(first, "content", "") or "").strip()
            checklist = _parse(raw)
            if len(checklist.steps) > self.plan_max_steps:
                checklist.steps = checklist.steps[: self.plan_max_steps]
            return checklist, warnings
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            warnings.append(f"planner parse failed; attempting repair: {exc}")
            try:
                repair = self.planner_llm.invoke(
                    [
                        SystemMessage(content=CODING_FLOW_PLANNER_PROMPT),
                        HumanMessage(
                            content=(
                                "Repair this into strict schema JSON only.\n\n"
                                f"Broken output:\n{raw if 'raw' in locals() else ''}\n\n"
                                f"Original request:\n{request}"
                            )
                        ),
                    ]
                )
                repaired_raw = str(getattr(repair, "content", "") or "").strip()
                checklist = _parse(repaired_raw)
                if len(checklist.steps) > self.plan_max_steps:
                    checklist.steps = checklist.steps[: self.plan_max_steps]
                return checklist, warnings
            except Exception as exc2:  # pragma: no cover - deterministic fallback
                warnings.append(f"planner repair failed: {exc2}")
                return None, warnings
        except Exception as exc:  # pragma: no cover
            warnings.append(f"planner invocation failed: {exc}")
            return None, warnings

    @staticmethod
    def _extract_payload(text: str) -> dict[str, Any] | None:
        raw = (text or "").strip()
        if not raw.startswith("{"):
            return None
        try:
            loaded = json.loads(raw)
        except Exception:
            return None
        return loaded if isinstance(loaded, dict) else None

    @staticmethod
    def _extract_answer_text(answer: str) -> str:
        payload = CodingAgent._extract_payload(answer)
        if payload is not None and isinstance(payload.get("answer"), str):
            return str(payload["answer"]).strip()
        return (answer or "").strip()

    @staticmethod
    def _tool_ok_from_trace_row(row: dict[str, Any]) -> bool | None:
        """
        Best-effort extraction of ok/failed from a tool trace row.
        Returns:
          - True/False if determinable
          - None if unknown
        """
        # Common patterns: { "status": "ok" } or { "ok": true } in the row itself
        if isinstance(row.get("ok"), bool):
            return bool(row["ok"])
        status = row.get("status")
        if isinstance(status, str):
            s = status.lower().strip()
            if s in ("ok", "success", "succeeded", "passed"):
                return True
            if s in ("error", "failed", "failure"):
                return False

        # Many agents store a JSON-ish preview under output_preview
        preview = row.get("output_preview")
        if isinstance(preview, str) and preview.strip().startswith("{"):
            try:
                payload = json.loads(preview)
                if isinstance(payload, dict) and isinstance(payload.get("ok"), bool):
                    return bool(payload["ok"])
            except Exception:
                pass

        return None

    def _patch_failure_summary(self, trace: list[dict[str, Any]]) -> tuple[int, int]:
        """
        Returns (apply_patch_calls, apply_patch_failures_detected)
        """
        calls = 0
        failures = 0
        for row in trace:
            if str(row.get("tool_name", "")) != "apply_patch":
                continue
            calls += 1
            ok = self._tool_ok_from_trace_row(row)
            if ok is False:
                failures += 1
        return calls, failures

    def _should_force_write_fallback(self, *, trace: list[dict[str, Any]], changed_files: list[str]) -> tuple[bool, str]:
        """
        Decide whether to force a write_file fallback based on observed apply_patch behavior.
        Trigger when:
        - apply_patch called >= threshold and no repo changes, OR
        - apply_patch failures detected >= threshold and no repo changes.
        """
        if changed_files:
            return False, ""
        patch_calls, patch_failures = self._patch_failure_summary(trace)
        threshold = int(self._PATCH_FAILURE_FALLBACK_THRESHOLD)
        if patch_calls >= threshold:
            return True, f"apply_patch attempted {patch_calls} times with no repo changes; forcing write_file fallback."
        if patch_failures >= threshold:
            return True, f"apply_patch failed {patch_failures} times with no repo changes; forcing write_file fallback."
        return False, ""

    def _compute_progress(
        self,
        *,
        checklist: FlowChecklist,
        trace: list[dict[str, Any]],
        warnings: list[str],
        changed_files: list[str],
        required_read_files: int,
    ) -> tuple[ExecutionDecision, dict[str, int], int]:
        counts: dict[str, int] = {}
        read_files: set[str] = set()
        for row in trace:
            name = str(row.get("tool_name", ""))
            counts[name] = counts.get(name, 0) + 1
            if name == "read_file":
                preview = str(row.get("output_preview", "") or "")
                payload = self._extract_payload(preview)
                if payload is not None:
                    path = str(payload.get("file_path", "")).strip()
                    if path:
                        read_files.add(path)

        read_count = len(read_files)
        if read_count < required_read_files:
            return (
                ExecutionDecision(
                    phase="blocked",
                    tool_call_allowed=False,
                    why=f"Need at least {required_read_files} unique read_file inspections before edit/answer.",
                ),
                counts,
                read_count,
            )

        if changed_files:
            phase = "verify" if any(x.endswith(".py") for x in changed_files) else "answer"
            return (
                ExecutionDecision(
                    phase=phase,
                    tool_call_allowed=True,
                    why="Edits detected; proceed to verification/answer.",
                ),
                counts,
                read_count,
            )

        # If patches are looping, do NOT push back to "inspect"; force edit (write_file fallback).
        force_fallback, reason = self._should_force_write_fallback(trace=trace, changed_files=changed_files)
        if force_fallback:
            return (
                ExecutionDecision(
                    phase="edit",
                    tool_call_allowed=True,
                    why=reason,
                ),
                counts,
                read_count,
            )

        if not changed_files:
            joined = "\n".join(warnings).lower()
            if "apply_patch failed" in joined or "patch did not apply cleanly" in joined or "patch-only loop" in joined:
                return (
                    ExecutionDecision(
                        phase="edit",
                        tool_call_allowed=True,
                        why="Patch failures detected via warnings with no repo changes; forcing write_file fallback.",
                    ),
                    counts,
                    read_count,
                )

        if warnings:
            return (
                ExecutionDecision(phase="inspect", tool_call_allowed=True, why="No edits yet; continue inspection."),
                counts,
                read_count,
            )

        _ = checklist
        return (
            ExecutionDecision(phase="edit", tool_call_allowed=True, why="Evidence gate met; editing allowed."),
            counts,
            read_count,
        )

    def _checklist_counts(self, checklist: FlowChecklist) -> dict[str, int]:
        done = len([step for step in checklist.steps if step.status == "done"])
        blocked = len([step for step in checklist.steps if step.status == "blocked"])
        pending = len(checklist.steps) - done - blocked
        return {"done": done, "pending": pending, "blocked": blocked, "total": len(checklist.steps)}

    def _tool_policy_for_request(self, request: str) -> dict[str, Any]:
        block_internet = self.repo_only_internet_default and (not self._allows_web_search(request))
        require_read_files = self.require_read_files
        explicit_files = {
            match.group(1).strip()
            for match in self._EXPLICIT_FILE_RE.finditer(request or "")
            if match.group(1).strip()
        }
        # Single-target file edits (e.g. README.md) should not be blocked by a 2-file gate.
        if len(explicit_files) == 1:
            require_read_files = 1
        return {
            "allowed_tools": [
                "semantic_search",
                "read_file",
                "run_command",
                "apply_patch",
                "write_file",
                "search_internet",
            ],
            "search_budget": self.search_budget,
            "read_budget": self.read_budget,
            "require_read_files": require_read_files,
            "block_internet": block_internet,
            "search_repeat_limit": 1,
            "max_semantic_k": 50,
        }

    def _generate_common(
        self,
        request: str,
        *,
        call_agent_fn,
        flow_context: str | None,
        flow_id: str | None,
    ) -> tuple[dict[str, Any], str | None, str | None]:
        before = self._git_status_paths()
        warnings: list[str] = []
        active_flow_id = flow_id
        effective_flow_context = flow_context
        if self.coding_memory_enabled and self.coding_memory_service is not None:
            try:
                active_flow_id = self.coding_memory_service.ensure_flow(flow_id=flow_id, request=request)
                self._current_flow_id = active_flow_id
                if effective_flow_context is None:
                    effective_flow_context = self.coding_memory_service.build_flow_context(
                        active_flow_id,
                        sorted(before),
                    )
            except Exception as exc:
                warnings.append(f"coding memory setup failed: {exc}")
                active_flow_id = flow_id

        checklist, plan_warnings = self._plan_checklist(request, flow_context=effective_flow_context)
        warnings.extend(plan_warnings)
        if checklist is None:
            result = {
                "status": "warning",
                "answer": "Planner failed to produce valid checklist JSON after repair; stopping to avoid blind tool loop.",
                "changed_files": [],
                "diff": "",
                "warnings": warnings,
                "flow_id": active_flow_id,
                "plan": None,
                "progress": {"phase": "blocked", "why": "planner_failed"},
                "checklist": {"done": 0, "pending": 0, "blocked": 1, "total": 0},
                "actions_taken": [],
                "next_step": "Refine request or retry planner with stricter constraints.",
            }
            return result, active_flow_id, effective_flow_context

        tool_policy = self._tool_policy_for_request(request)
        required_read_files = int(tool_policy.get("require_read_files", self.require_read_files) or self.require_read_files)

        # First pass
        answer = call_agent_fn(tool_policy=tool_policy, flow_context=effective_flow_context)

        after = self._git_status_paths()
        changed = sorted(after.difference(before))

        payload = self._extract_payload(answer) or {}
        trace = payload.get("trace", [])
        if not isinstance(trace, list):
            trace = []

        # If patch loop detected and no changes, force a single retry without apply_patch.
        force_fallback, force_reason = self._should_force_write_fallback(
            trace=[item for item in trace if isinstance(item, dict)],
            changed_files=changed,
        )
        if force_fallback:
            warnings.append(force_reason)
            logger.warning(force_reason)

            # Nudge the model strongly toward write_file chunk/finalize.
            retry_request = (
                "NOTE: apply_patch is unavailable or failed/no-op repeatedly. "
                "Use write_file fallback (chunked part_index + finalize) with full file content.\n\n"
                f"Original request:\n{request}"
            )

            # Disable apply_patch tool for this retry so the agent must use write_file (or read/run).
            with self._without_tool("apply_patch"):
                answer = call_agent_fn(tool_policy=tool_policy, flow_context=effective_flow_context)

            after2 = self._git_status_paths()
            changed = sorted(after2.difference(before))

            # Refresh payload/trace from retry for reporting
            payload = self._extract_payload(answer) or {}
            trace = payload.get("trace", [])
            if not isinstance(trace, list):
                trace = []

        findings = self._run_static_analysis([p for p in changed if p.endswith(".py")])
        diff = self._git_diff(changed)
        answer_text = self._extract_answer_text(answer)

        payload_warnings = payload.get("warnings", [])
        if isinstance(payload_warnings, list):
            warnings.extend(str(item) for item in payload_warnings if str(item).strip())

        decision, counters, read_count = self._compute_progress(
            checklist=checklist,
            trace=[item for item in trace if isinstance(item, dict)],
            warnings=warnings,
            changed_files=changed,
            required_read_files=required_read_files,
        )
        force_write = (
        decision.phase == "edit"
        and "forcing write_file fallback" in decision.why.lower()
        and not changed
        )

        if force_write:
            warnings.append("Forcing write_file fallback: re-running agent with apply_patch disabled.")
            logger.warning("Forcing write_file fallback: re-running agent with apply_patch disabled.")

            with self._without_tool("apply_patch"):
                answer = call_agent_fn(tool_policy=tool_policy, flow_context=effective_flow_context)

            after2 = self._git_status_paths()
            changed = sorted(after2.difference(before))

            # refresh diff/findings/trace based on retry output
            findings = self._run_static_analysis([p for p in changed if p.endswith(".py")])
            diff = self._git_diff(changed)

            payload = self._extract_payload(answer) or {}
            trace = payload.get("trace", [])
            if not isinstance(trace, list):
                trace = []
        checklist_counts = self._checklist_counts(checklist)
        transitions = [
            {
                "from_phase": "discover",
                "to_phase": decision.phase,
                "reason": decision.why,
            }
        ]
        effective_prompt = self._effective_system_prompt_for(request, flow_context=effective_flow_context)

        if (
            self.coding_memory_enabled
            and self.coding_memory_service is not None
            and active_flow_id is not None
        ):
            try:
                self.coding_memory_service.record_turn(
                    flow_id=active_flow_id,
                    user_request=request,
                    effective_prompt=effective_prompt,
                    agent_answer=answer_text,
                    changed_files=changed,
                    warnings=warnings,
                    static_findings=[_as_jsonable(f) for f in findings],
                    checklist=checklist.model_dump(),
                    transitions=transitions,
                )
            except Exception as exc:
                warnings.append(f"coding memory turn persistence failed: {exc}")

        status = "ok" if not findings else "warning"
        if decision.phase == "blocked":
            status = "warning"
        result = {
            "status": status,
            "answer": answer,
            "changed_files": changed,
            "diff": diff,
            "warnings": warnings,
            "flow_id": active_flow_id,
            "plan": checklist.model_dump(),
            "progress": {
                "phase": decision.phase,
                "why": decision.why,
                "budgets": {
                    "search_budget": self.search_budget,
                    "search_used": counters.get("semantic_search", 0),
                    "read_budget": self.read_budget,
                    "read_used": counters.get("read_file", 0),
                    "required_read_files": required_read_files,
                    "read_files_observed": read_count,
                },
            },
            "checklist": checklist_counts,
            "actions_taken": [item for item in trace if isinstance(item, dict)][:20],
            "next_step": checklist.next_action or decision.why,
            "static_analysis": {
                "finding_count": len(findings),
                "findings": [_as_jsonable(f) for f in findings],
            },
        }
        return result, active_flow_id, effective_flow_context

    def generate(
        self,
        request: str,
        *,
        index_dir: str | Path | None = None,
        k: int | None = None,
        max_steps: int = 200,
        timeout_seconds: int = 600,
        callbacks: Sequence[Any] | None = None,
        flow_context: str | None = None,
        flow_id: str | None = None,
    ) -> dict[str, Any]:
        def _call(*, tool_policy: dict[str, Any], flow_context: str | None) -> str:
            return self._call_agent_single(
                request,
                index_dir=index_dir,
                k=k,
                max_steps=max_steps,
                timeout_seconds=timeout_seconds,
                callbacks=callbacks,
                flow_context=flow_context,
                tool_policy=tool_policy,
            )

        result, _, _ = self._generate_common(
            request,
            call_agent_fn=_call,
            flow_context=flow_context,
            flow_id=flow_id,
        )
        return result

    def generate_dir_mode(
        self,
        request: str,
        *,
        index_dirs: Sequence[str | Path],
        k: int | None = None,
        max_steps: int = 200,
        timeout_seconds: int = 600,
        callbacks: Sequence[Any] | None = None,
        flow_context: str | None = None,
        flow_id: str | None = None,
    ) -> dict[str, Any]:
        def _call(*, tool_policy: dict[str, Any], flow_context: str | None) -> str:
            return self._call_agent_multi(
                request,
                index_dirs=index_dirs,
                k=k,
                max_steps=max_steps,
                timeout_seconds=timeout_seconds,
                callbacks=callbacks,
                flow_context=flow_context,
                tool_policy=tool_policy,
            )

        result, _, _ = self._generate_common(
            request,
            call_agent_fn=_call,
            flow_context=flow_context,
            flow_id=flow_id,
        )
        return result

    def _call_agent_single(
        self,
        request: str,
        *,
        index_dir: str | Path | None,
        k: int | None,
        max_steps: int,
        timeout_seconds: int,
        callbacks: Sequence[Any] | None,
        flow_context: str | None = None,
        tool_policy: dict[str, Any] | None = None,
    ) -> str:
        effective_prompt = self._effective_system_prompt_for(request, flow_context=flow_context)
        if hasattr(self.ask_agent, "run"):
            return self._stringify(
                self._invoke_run_like(
                    "run",
                    question=request,
                    index_dir=str(Path(index_dir).resolve()) if index_dir is not None else None,
                    k=k,
                    max_steps=max_steps,
                    timeout_seconds=timeout_seconds,
                    callbacks=callbacks,
                    system_prompt=effective_prompt,
                    tool_policy=tool_policy,
                )
            )
        return self._call_ask_like(request, system_prompt=effective_prompt)

    def _call_agent_multi(
        self,
        request: str,
        *,
        index_dirs: Sequence[str | Path],
        k: int | None,
        max_steps: int,
        timeout_seconds: int,
        callbacks: Sequence[Any] | None,
        flow_context: str | None = None,
        tool_policy: dict[str, Any] | None = None,
    ) -> str:
        resolved = [str(Path(p).resolve()) for p in index_dirs if str(p).strip()]
        if not resolved:
            return "No index_dirs provided for dir-mode."
        effective_prompt = self._effective_system_prompt_for(request, flow_context=flow_context)
        if hasattr(self.ask_agent, "run_multi"):
            return self._stringify(
                self._invoke_run_like(
                    "run_multi",
                    question=request,
                    index_dirs=resolved,
                    k=k,
                    max_steps=max_steps,
                    timeout_seconds=timeout_seconds,
                    callbacks=callbacks,
                    system_prompt=effective_prompt,
                    tool_policy=tool_policy,
                )
            )
        if hasattr(self.ask_agent, "run"):
            stitched = (
                f"{request}\n\n"
                "DIR-MODE CONTEXT:\n"
                f"- index_dirs:\n  - " + "\n  - ".join(resolved) + "\n"
                "- Use these indexes when searching context.\n"
            )
            return self._stringify(
                self._invoke_run_like(
                    "run",
                    question=stitched,
                    index_dir=resolved[0],
                    k=k,
                    max_steps=max_steps,
                    timeout_seconds=timeout_seconds,
                    callbacks=callbacks,
                    system_prompt=effective_prompt,
                    tool_policy=tool_policy,
                )
            )
        return self._call_ask_like(request, system_prompt=effective_prompt)

    @contextmanager
    def _without_tool(self, tool_name: str):
        tools = getattr(self.ask_agent, "tools", None)
        if not isinstance(tools, list):
            yield
            return
        original = list(tools)
        tools[:] = [tool for tool in tools if str(getattr(tool, "name", "")) != tool_name]
        try:
            yield
        finally:
            tools[:] = original

    def _call_ask_like(self, request: str, *, system_prompt: str) -> str:
        kwargs: dict[str, Any] = {}
        try:
            sig = inspect.signature(self.ask_agent.ask)
            if "tool_use" in sig.parameters:
                kwargs["tool_use"] = True
            if "system_prompt" in sig.parameters:
                kwargs["system_prompt"] = system_prompt
            elif "instructions" in sig.parameters:
                kwargs["instructions"] = system_prompt
        except Exception:
            pass
        if "system_prompt" not in kwargs and "instructions" not in kwargs:
            request = f"{system_prompt}\n\nUser request:\n{request}"
        result = self.ask_agent.ask(request, **kwargs)
        return self._stringify(result)

    def _invoke_run_like(self, method_name: str, *, system_prompt: str, **args: Any) -> Any:
        fn = getattr(self.ask_agent, method_name)
        try:
            sig = inspect.signature(fn)
            filtered: dict[str, Any] = {}
            for k, v in args.items():
                if v is None:
                    continue
                if k in sig.parameters:
                    filtered[k] = v
            if "system_prompt" in sig.parameters:
                filtered["system_prompt"] = system_prompt
            elif "instructions" in sig.parameters:
                filtered["instructions"] = system_prompt
            return fn(**filtered)
        except Exception:
            try:
                return fn(args.get("question"))
            except Exception:
                raise

    def _stringify(self, result: Any) -> str:
        if isinstance(result, str):
            return result
        try:
            return json.dumps(_as_jsonable(result), indent=2, ensure_ascii=False)
        except Exception:
            return str(result)

    def _git_status_paths(self) -> set[str]:
        try:
            proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                return set()
            paths: set[str] = set()
            for line in proc.stdout.splitlines():
                if len(line) >= 4:
                    p = line[3:].strip()
                    if p:
                        paths.add(p.replace("\\", "/"))
            return paths
        except Exception:
            return set()

    def _git_diff(self, paths: list[str]) -> str:
        if not paths:
            return ""
        try:
            proc = subprocess.run(
                ["git", "diff", "--", *paths],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                return ""
            return proc.stdout[:200_000]
        except Exception:
            return ""

    def _run_static_analysis(self, py_paths: list[str]) -> list[Any]:
        if not py_paths:
            return []
        try:
            from mana_analyzer.analysis.checks import PythonStaticAnalyzer  # type: ignore

            analyzer = PythonStaticAnalyzer()
            all_findings: list[Any] = []
            for rel in py_paths:
                p = (self.repo_root / rel).resolve()
                try:
                    findings = analyzer.analyze_file(p)
                    if findings:
                        all_findings.extend(findings)
                except Exception as exc:
                    all_findings.append({"path": str(rel), "error": f"Static analysis error: {exc}"})
            return all_findings
        except Exception as exc:
            logger.debug("Static analysis unavailable: %s", exc)
            return []

    def get_active_flow_id(self) -> str | None:
        if self._current_flow_id:
            return self._current_flow_id
        if not self.coding_memory_enabled or self.coding_memory_service is None:
            return None
        return self.coding_memory_service.get_active_flow_id()

    def flow_summary(self, flow_id: str | None = None) -> dict[str, Any] | None:
        if not self.coding_memory_enabled or self.coding_memory_service is None:
            return None
        target = flow_id or self.get_active_flow_id()
        if not target:
            return None
        summary = self.coding_memory_service.get_flow_summary(target)
        if summary is None:
            return None
        return {
            "flow_id": summary.flow_id,
            "objective": summary.objective,
            "updated_at": summary.updated_at,
            "constraints": summary.constraints,
            "acceptance": summary.acceptance,
            "open_tasks": summary.open_tasks,
            "recent_decisions": summary.recent_decisions,
            "last_changed_files": summary.last_changed_files,
            "unresolved_static_findings": summary.unresolved_static_findings,
            "checklist": summary.checklist,
            "transitions": summary.transitions,
            "last_blocked_reason": summary.last_blocked_reason,
            "recent_turns": self.coding_memory_service.list_recent_turns(summary.flow_id),
        }

    def reset_flow(self, flow_id: str | None = None) -> str | None:
        if not self.coding_memory_enabled or self.coding_memory_service is None:
            self._current_flow_id = None
            return None
        target = flow_id or self.get_active_flow_id()
        if not target:
            self._current_flow_id = None
            return None
        self.coding_memory_service.reset_flow(target)
        self._current_flow_id = None
        return target

    def checkpoint_flow(self, flow_id: str | None = None) -> str | None:
        if not self.coding_memory_enabled or self.coding_memory_service is None:
            return None
        target = flow_id or self.get_active_flow_id()
        if not target:
            return None
        summary = self.flow_summary(target) or {}
        self.coding_memory_service.checkpoint(target, snapshot=summary)
        return target

    def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
        if not self.coding_memory_enabled or self.coding_memory_service is None:
            return False
        target = flow_id or self.get_active_flow_id()
        if not target:
            return False
        return self.coding_memory_service.is_conflicting_request(target, request)