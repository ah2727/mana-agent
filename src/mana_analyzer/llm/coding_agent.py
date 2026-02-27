"""
mana_analyzer.llm.coding_agent

Coding agent wrapper with:
- mutation tools (write_file/apply_patch)
- structured flow/checklist planning
- anti-loop tool policy (search/read budgets, duplicate search guards)
- flow-memory continuity integration
"""

from __future__ import annotations

import ast
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
    FULL_AUTO_EXECUTION_PROMPT,
)
from mana_analyzer.llm.tool_worker_process import (
    ToolRunRequest,
    ToolWorkerClient,
    ToolWorkerProcessError,
)
from mana_analyzer.llm.tools_manager import ToolsManagerOrchestrator
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
- You may modify any file under the repository root when needed for the request.
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
3) apply_patch internally tries a fallback chain:
   - git apply
   - perl fallback applier
   - python fallback compute
   - write_file persistence
4) After each mutation attempt (apply_patch/write_file), verify file-change evidence (changed_files, git status, or diff).
5) If mutation reports success but files did not change, treat as no-op and retry with corrected patch/full content.
6) If apply_patch still fails twice OR repeated mutation attempts produce no repo changes, STOP patch-only loops and use explicit write_file fallback:
   - read_file the target
   - generate full updated file content
   - write_file in chunks with part_index, then finalize=true
7) Keep retries bounded by anti-loop policy; if no-op persists after bounded retries, return blocked status with concrete reason.
8) If user intent is an edit and you already know the target file/content change, execute the mutation in this turn and do not emit "if you want me to proceed" style confirmation text.

"""


class FlowStep(BaseModel):
    """Represents a planned step with tooling guidance and execution status."""

    id: str
    title: str
    reason: str
    status: Literal["pending", "in_progress", "done", "blocked"] = "pending"
    requires_tools: list[str] = Field(default_factory=list)


class FlowChecklist(BaseModel):
    """Structured plan capturing the objective, constraints, acceptance criteria, and steps."""

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
    _AMBIGUOUS_FOLLOWUP_RE = re.compile(
        r"(?i)^\s*(yes|yep|ok|okay|sure|continue|go|proceed|begin|start|do it|done|next)\s*[/!.]?\s*$"
    )
    _PLAN_TRIGGER_RE = re.compile(
        r"(?i)^\s*(?:please\s+)?(?:implement|execute|run|apply|trigger)\s+"
        r"(?:the\s+|last\s+|that\s+|current\s+)?plan\s*[/!.]?\s*$"
    )

    # Fallback policy: if apply_patch is attempted >= this many times and no changes appear, force write_file fallback.
    _PATCH_FAILURE_FALLBACK_THRESHOLD = 2

    @staticmethod
    def _parse_json_or_literal(raw: str) -> Any | None:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        try:
            return ast.literal_eval(text)
        except Exception:
            return None

    @staticmethod
    def _checklist_from_plan_text(text: str, request: str = "") -> FlowChecklist | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        if "\\n" in raw and "\n" not in raw:
            raw = raw.replace("\\n", "\n")
        steps: list[str] = []
        objective = ""
        for line in raw.splitlines():
            item = line.strip()
            if not item:
                continue
            lowered = item.lower()
            if lowered.startswith("objective:"):
                objective = item.split(":", 1)[1].strip()
                continue
            if lowered in {"plan:", "execution plan:", "**execution plan**"}:
                continue
            match = re.match(r"^(?:\d+[.)]\s+|[-*]\s+)(.+)$", item)
            if match:
                step_text = match.group(1).strip()
                if step_text:
                    steps.append(step_text[:220])
        if not steps:
            return None
        resolved_objective = objective or (" ".join((request or "").strip().split())[:220] or "Implement requested change")
        flow_steps: list[FlowStep] = []
        for idx, title in enumerate(steps[:20], start=1):
            flow_steps.append(
                FlowStep(
                    id=f"s{idx}",
                    title=title,
                    reason="Derived from planner text output",
                    status="in_progress" if idx == 1 else "pending",
                    requires_tools=[],
                )
            )
        return FlowChecklist(
            objective=resolved_objective,
            constraints=[],
            acceptance=["Requested change is applied"],
            steps=flow_steps,
            next_action=flow_steps[0].title if flow_steps else "",
        )

    @classmethod
    def _coerce_checklist_from_obj(cls, parsed: Any, request: str = "") -> FlowChecklist | None:
        if isinstance(parsed, dict):
            if "objective" in parsed and "steps" in parsed:
                return FlowChecklist.model_validate(parsed)
            text_payload = parsed.get("text")
            if isinstance(text_payload, str):
                return cls._checklist_from_plan_text(text_payload, request=request)
            return None
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "objective" in item and "steps" in item:
                    return FlowChecklist.model_validate(item)
            text_chunks: list[str] = []
            for item in parsed:
                if isinstance(item, dict):
                    text_value = item.get("text")
                    if isinstance(text_value, str) and text_value.strip():
                        text_chunks.append(text_value.strip())
            if text_chunks:
                return cls._checklist_from_plan_text("\n".join(text_chunks), request=request)
        return None

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        raw = str(text or "").strip()
        if not raw.startswith("```"):
            return raw
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
        return raw

    @classmethod
    def _collect_checklist_candidates(cls, raw_text: str) -> list[Any]:
        pending: list[Any] = [raw_text]
        candidates: list[Any] = []
        seen_text: set[str] = set()
        seen_ids: set[int] = set()

        while pending:
            item = pending.pop(0)
            if isinstance(item, str):
                text = item.strip()
                if not text or text in seen_text:
                    continue
                seen_text.add(text)
                candidates.append(text)

                unwrapped = cls._strip_code_fence(text)
                if unwrapped and unwrapped not in seen_text:
                    pending.append(unwrapped)

                obj_text = cls._extract_json_object_text(text)
                if obj_text and obj_text not in seen_text:
                    pending.append(obj_text)

                parsed = cls._parse_json_or_literal(text)
                if parsed is not None:
                    pending.append(parsed)
                continue

            if isinstance(item, dict):
                marker = id(item)
                if marker in seen_ids:
                    continue
                seen_ids.add(marker)
                candidates.append(item)

                for key in ("answer", "content", "text", "message", "output", "payload", "data", "raw"):
                    if key in item:
                        pending.append(item.get(key))
                for value in item.values():
                    if isinstance(value, (dict, list)):
                        pending.append(value)
                    elif isinstance(value, str) and len(value) <= 20000:
                        pending.append(value)
                continue

            if isinstance(item, list):
                marker = id(item)
                if marker in seen_ids:
                    continue
                seen_ids.add(marker)
                candidates.append(item)
                pending.extend(item)

        return candidates

    @staticmethod
    def _extract_json_object_text(text: str) -> str | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        raw = CodingAgent._strip_code_fence(raw)
        if raw.startswith("{") and raw.endswith("}"):
            return raw
        start = raw.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(raw)):
            ch = raw[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw[start : idx + 1].strip()
                    return candidate if candidate else None
        return None

    @classmethod
    def _parse_flow_checklist_json(cls, text: str, request: str = "") -> FlowChecklist:
        raw = str(text or "").strip()
        for candidate in cls._collect_checklist_candidates(raw):
            if isinstance(candidate, (dict, list)):
                try:
                    checklist = cls._coerce_checklist_from_obj(candidate, request=request)
                except (ValidationError, TypeError, ValueError):
                    checklist = None
                if checklist is not None:
                    return checklist
                continue

            if not isinstance(candidate, str):
                continue

            parsed_candidate = cls._parse_json_or_literal(candidate)
            if parsed_candidate is not None:
                try:
                    checklist = cls._coerce_checklist_from_obj(parsed_candidate, request=request)
                except (ValidationError, TypeError, ValueError):
                    checklist = None
                if checklist is not None:
                    return checklist

            text_checklist = cls._checklist_from_plan_text(candidate, request=request)
            if text_checklist is not None:
                return text_checklist

        raise json.JSONDecodeError("No checklist payload found", raw, 0)

    def _fallback_checklist(self, request: str) -> FlowChecklist:
        explicit_files = sorted(
            {
                match.group(1).strip()
                for match in self._EXPLICIT_FILE_RE.finditer(request or "")
                if match.group(1).strip()
            }
        )
        objective = " ".join((request or "").strip().split())[:220] or "Implement requested change"
        inspect_title = "Inspect target file(s)" if explicit_files else "Discover target file(s)"
        inspect_reason = (
            f"Validate current behavior in: {', '.join(explicit_files[:3])}"
            if explicit_files
            else "Collect concrete evidence before edits"
        )
        return FlowChecklist(
            objective=objective,
            acceptance=["Requested change is applied", "No obvious regressions in touched files"],
            steps=[
                FlowStep(
                    id="s1",
                    title=inspect_title,
                    reason=inspect_reason,
                    status="in_progress",
                    requires_tools=["semantic_search", "read_file"],
                ),
                FlowStep(
                    id="s2",
                    title="Apply requested change",
                    reason="Implement the user request in repository files",
                    status="pending",
                    requires_tools=["apply_patch", "write_file"],
                ),
                FlowStep(
                    id="s3",
                    title="Verify and summarize",
                    reason="Confirm edits and report outcomes",
                    status="pending",
                    requires_tools=["run_command", "read_file"],
                ),
            ],
            next_action="Inspect file context, then apply the requested edit.",
        )

    def __init__(
        self,
        *,
        api_key: str,
        repo_root: Path,
        ask_agent: AskAgentLike,
        base_url: str | None = None,
        allowed_prefixes: Optional[Sequence[str]] = ("src/", "tests/", ""),
        system_prompt: str = CODING_SYSTEM_PROMPT,
        coding_memory_service: CodingMemoryService | None = None,
        coding_memory_enabled: bool = True,
        plan_max_steps: int = 8,
        search_budget: int = 4,
        read_budget: int = 6,
        require_read_files: int = 2,
        repo_only_internet_default: bool = True,
        tool_worker_client: ToolWorkerClient | None = None,
        full_auto_mode: bool = False,
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
        self.tool_worker_client = tool_worker_client
        self.full_auto_mode = bool(full_auto_mode)
        self.tools_manager_orchestrator: ToolsManagerOrchestrator | None = None

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

    def set_tools_manager_orchestrator(self, orchestrator: ToolsManagerOrchestrator | None) -> None:
        self.tools_manager_orchestrator = orchestrator

    @staticmethod
    def _normalize_prechecklist(checklist: FlowChecklist, *, source: str) -> dict[str, Any]:
        steps: list[dict[str, str]] = []
        for item in checklist.steps[:20]:
            steps.append(
                {
                    "id": str(item.id or "").strip() or "step",
                    "title": str(item.title or "").strip() or "step",
                    "status": str(item.status or "pending"),
                }
            )
        return {
            "objective": str(checklist.objective or "").strip(),
            "steps": steps,
            "source": str(source or ""),
        }

    def preview_execution_checklist(
        self,
        request: str,
        *,
        flow_id: str | None = None,
        flow_context: str | None = None,
    ) -> dict[str, Any]:
        """Build and persist a pre-execution checklist preview for UI rendering."""
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

        checklist, plan_warnings, source = self._plan_checklist_with_source(
            request,
            flow_context=effective_flow_context,
        )
        warnings.extend(plan_warnings)
        if checklist is None:
            return {
                "flow_id": active_flow_id,
                "flow_context": effective_flow_context,
                "prechecklist": {"objective": "", "steps": [], "source": "deterministic_fallback"},
                "prechecklist_source": "deterministic_fallback",
                "prechecklist_warning": "Planner failed to produce a preview checklist.",
                "warnings": warnings,
            }

        prechecklist = self._normalize_prechecklist(checklist, source=source)
        prechecklist_warning = ""
        if source == "deterministic_fallback":
            prechecklist_warning = "Planner parse failed; using deterministic fallback checklist."

        if (
            self.coding_memory_enabled
            and self.coding_memory_service is not None
            and active_flow_id is not None
        ):
            try:
                self.coding_memory_service.persist_preview_checklist(
                    flow_id=active_flow_id,
                    user_request=request,
                    checklist=checklist.model_dump(),
                    source=source,
                    warning=prechecklist_warning,
                )
            except Exception as exc:
                warnings.append(f"coding memory preview persistence failed: {exc}")

        return {
            "flow_id": active_flow_id,
            "flow_context": effective_flow_context,
            "prechecklist": prechecklist,
            "prechecklist_source": source,
            "prechecklist_warning": prechecklist_warning,
            "warnings": warnings,
        }

    def generate_auto_execute(
        self,
        request: str,
        *,
        index_dir: str | Path | None = None,
        index_dirs: Sequence[str | Path] | None = None,
        k: int | None = None,
        max_steps: int = 200,
        timeout_seconds: int = 600,
        pass_cap: int = 4,
        flow_context: str | None = None,
        flow_id: str | None = None,
        callbacks: Sequence[Any] | None = None,
        prechecklist_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        preview_payload = prechecklist_payload or self.preview_execution_checklist(
            request,
            flow_id=flow_id,
            flow_context=flow_context,
        )
        prechecklist = preview_payload.get("prechecklist") if isinstance(preview_payload.get("prechecklist"), dict) else None
        prechecklist_source = str(preview_payload.get("prechecklist_source", "") or "")
        prechecklist_warning = str(preview_payload.get("prechecklist_warning", "") or "")

        if self.tools_manager_orchestrator is None:
            return {
                "status": "warning",
                "answer": "Auto-execute orchestrator is unavailable for this session.",
                "changed_files": [],
                "diff": "",
                "warnings": ["auto_execute_orchestrator_unavailable"],
                "flow_id": flow_id,
                "plan": None,
                "progress": {"phase": "blocked", "why": "auto_execute_orchestrator_unavailable"},
                "checklist": {"done": 0, "pending": 0, "blocked": 1, "total": 0},
                "actions_taken_total": 0,
                "actions_taken_truncated": False,
                "actions_taken": [],
                "next_step": "Initialize tool worker and tools manager orchestrator, then retry.",
                "static_analysis": {"finding_count": 0, "findings": []},
                "auto_execute_passes": 0,
                "auto_execute_terminal_reason": "orchestrator_unavailable",
                "toolsmanager_requests_count": 0,
                "pass_logs": [],
                "planner_decisions": [],
                "tool_execution_backend": "",
                "tool_execution_run_id": "",
                "tool_execution_duration_ms": 0.0,
                "tool_execution_requests_ok": 0,
                "tool_execution_requests_failed": 0,
                "prechecklist": prechecklist,
                "prechecklist_source": prechecklist_source,
                "prechecklist_warning": prechecklist_warning,
            }

        before = self._git_status_paths()
        warnings: list[str] = []
        preview_warnings = preview_payload.get("warnings") if isinstance(preview_payload.get("warnings"), list) else []
        warnings.extend(str(item).strip() for item in preview_warnings if str(item).strip())
        active_flow_id = flow_id
        effective_flow_context = flow_context
        if isinstance(preview_payload.get("flow_id"), str) and str(preview_payload.get("flow_id")).strip():
            active_flow_id = str(preview_payload.get("flow_id")).strip()
        if effective_flow_context is None and isinstance(preview_payload.get("flow_context"), str):
            flow_ctx = str(preview_payload.get("flow_context") or "").strip()
            effective_flow_context = flow_ctx or None
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

        tool_policy = self._tool_policy_for_request(request)
        try:
            _ = callbacks
            orchestrated = self.tools_manager_orchestrator.run(
                request=request,
                flow_context=effective_flow_context,
                index_dir=index_dir,
                index_dirs=index_dirs,
                k=int(k if k is not None else 8),
                max_steps=int(max_steps),
                timeout_seconds=int(timeout_seconds),
                tool_policy=tool_policy,
                pass_cap=max(1, int(pass_cap)),
                on_event=self._log_worker_event,
            )
        except ToolWorkerProcessError as exc:
            warnings.append(f"toolsmanager worker failure: {exc.code}: {exc}")
            return {
                "status": "warning",
                "answer": "Auto-execute failed due to worker process error.",
                "changed_files": [],
                "diff": "",
                "warnings": warnings,
                "flow_id": active_flow_id,
                "plan": None,
                "progress": {"phase": "blocked", "why": "tool_worker_error"},
                "checklist": {"done": 0, "pending": 0, "blocked": 1, "total": 0},
                "actions_taken_total": 0,
                "actions_taken_truncated": False,
                "actions_taken": [],
                "next_step": "Retry with a more specific tool-executable request.",
                "static_analysis": {"finding_count": 0, "findings": []},
                "auto_execute_passes": 0,
                "auto_execute_terminal_reason": "tool_worker_error",
                "toolsmanager_requests_count": 0,
                "pass_logs": [],
                "planner_decisions": [],
                "tool_execution_backend": "",
                "tool_execution_run_id": "",
                "tool_execution_duration_ms": 0.0,
                "tool_execution_requests_ok": 0,
                "tool_execution_requests_failed": 0,
                "prechecklist": prechecklist,
                "prechecklist_source": prechecklist_source,
                "prechecklist_warning": prechecklist_warning,
            }

        changed_files = sorted(self._git_status_paths().difference(before))
        changed_for_result = sorted({*changed_files, *list(orchestrated.changed_files)})
        findings = self._run_static_analysis([p for p in changed_for_result if p.endswith(".py")])
        diff = self._git_diff(changed_for_result)
        warnings.extend([str(item).strip() for item in orchestrated.warnings if str(item).strip()])
        checklist_total = len((orchestrated.plan or {}).get("steps", []) if isinstance(orchestrated.plan, dict) else [])
        checklist_done = checklist_total if changed_for_result else 0

        planner_decisions = (
            list(orchestrated.planner_decisions)
            if isinstance(orchestrated.planner_decisions, list)
            else []
        )

        if (
            self.coding_memory_enabled
            and self.coding_memory_service is not None
            and active_flow_id is not None
        ):
            try:
                transitions: list[dict[str, Any]] = []
                for item in planner_decisions:
                    if not isinstance(item, dict):
                        continue
                    transitions.append(
                        {
                            "from_phase": f"pass_{int(item.get('pass_index', 0) or 0)}",
                            "to_phase": str(item.get("decision", "continue") or "continue"),
                            "reason": str(item.get("decision_reason", "") or "").strip()
                            or "planner decision",
                        }
                    )
                transitions.append(
                    {
                        "from_phase": "auto_execute",
                        "to_phase": "answer",
                        "reason": f"auto_execute_terminal_reason={orchestrated.terminal_reason}",
                    }
                )
                self.coding_memory_service.record_turn(
                    flow_id=active_flow_id,
                    user_request=request,
                    effective_prompt=self._effective_system_prompt_for(request, flow_context=effective_flow_context),
                    agent_answer=str(orchestrated.answer or ""),
                    changed_files=changed_for_result,
                    warnings=warnings,
                    static_findings=[_as_jsonable(f) for f in findings],
                    checklist=orchestrated.plan or {},
                    transitions=transitions,
                )
            except Exception as exc:
                warnings.append(f"coding memory turn persistence failed: {exc}")

        status = "ok" if not findings else "warning"
        return {
            "status": status,
            "answer": str(orchestrated.answer or ""),
            "changed_files": changed_for_result,
            "diff": diff,
            "warnings": warnings,
            "flow_id": active_flow_id,
            "plan": orchestrated.plan,
            "progress": {
                "phase": "answer",
                "why": f"auto_execute_terminal_reason={orchestrated.terminal_reason}",
                "budgets": {
                    "search_budget": self.search_budget,
                    "read_budget": self.read_budget,
                    "required_read_files": int(tool_policy.get("require_read_files", self.require_read_files) or self.require_read_files),
                },
            },
            "checklist": {
                "done": checklist_done,
                "pending": max(0, checklist_total - checklist_done),
                "blocked": 0,
                "total": checklist_total,
            },
            "actions_taken_total": len(orchestrated.trace),
            "actions_taken_truncated": len(orchestrated.trace) > 20,
            "actions_taken": orchestrated.trace[:20],
            "next_step": orchestrated.terminal_reason or "Completed.",
            "static_analysis": {
                "finding_count": len(findings),
                "findings": [_as_jsonable(f) for f in findings],
            },
            "auto_execute_passes": orchestrated.passes,
            "auto_execute_terminal_reason": orchestrated.terminal_reason,
            "toolsmanager_requests_count": orchestrated.toolsmanager_requests_count,
            "pass_logs": orchestrated.pass_logs,
            "planner_decisions": planner_decisions,
            "tool_execution_backend": orchestrated.execution_backend,
            "tool_execution_run_id": orchestrated.execution_run_id,
            "tool_execution_duration_ms": orchestrated.execution_duration_ms,
            "tool_execution_requests_ok": orchestrated.execution_requests_ok,
            "tool_execution_requests_failed": orchestrated.execution_requests_failed,
            "prechecklist": prechecklist,
            "prechecklist_source": prechecklist_source,
            "prechecklist_warning": prechecklist_warning,
        }

    def _looks_like_edit_request(self, request: str) -> bool:
        lowered = request.lower()
        return any(token in lowered for token in self._EDIT_INTENT_TOKENS)

    @staticmethod
    def _looks_like_conversational_terminal(text: str) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return False
        patterns = (
            "if you want",
            "reply \"yes",
            "reply 'yes",
            "let me know if you want",
            "if you want, i can",
            "if you want i can",
            "say yes",
            "type yes",
        )
        return any(token in lowered for token in patterns)

    def _allows_web_search(self, request: str) -> bool:
        lowered = request.lower()
        return any(token in lowered for token in self._WEB_INTENT_TOKENS)

    @classmethod
    def _is_ambiguous_followup(cls, request: str) -> bool:
        text = (request or "").strip()
        if not text:
            return False
        if cls._PLAN_TRIGGER_RE.match(text):
            return True
        if cls._AMBIGUOUS_FOLLOWUP_RE.match(text):
            return True
        # Keep very short acknowledgements as ambiguous unless they include file/symbol hints.
        if len(text) <= 12 and not cls._EXPLICIT_FILE_RE.search(text):
            lowered = text.lower()
            return lowered in {"yes", "yes.", "ok", "ok.", "begin", "begin.", "go", "go.", "continue", "continue."}
        return False

    @classmethod
    def _is_plan_trigger_followup(cls, request: str) -> bool:
        return bool(cls._PLAN_TRIGGER_RE.match((request or "").strip()))

    @staticmethod
    def _extract_objective_from_flow_context(flow_context: str | None) -> str:
        context = (flow_context or "").strip()
        if not context:
            return ""
        for line in context.splitlines():
            text = line.strip()
            if text.lower().startswith("current objective:"):
                return text.split(":", 1)[1].strip()
        return ""

    @staticmethod
    def _extract_pending_steps_from_flow_context(flow_context: str | None) -> list[str]:
        context = (flow_context or "").strip()
        if not context:
            return []
        pending: list[str] = []
        for line in context.splitlines():
            text = line.strip()
            match = re.match(r"^- \[(pending|in_progress)\]\s+(.+)$", text, flags=re.IGNORECASE)
            if match:
                title = match.group(2).strip()
                if title:
                    pending.append(title[:220])
        return pending[:8]

    def _rewrite_ambiguous_followup(self, request: str, flow_context: str | None) -> str:
        if not self._is_ambiguous_followup(request):
            return request
        objective = self._extract_objective_from_flow_context(flow_context)
        if not objective:
            return request
        if self._is_plan_trigger_followup(request):
            pending_steps = self._extract_pending_steps_from_flow_context(flow_context)
            pending_block = ""
            if pending_steps:
                lines = "\n".join(f"- {item}" for item in pending_steps)
                pending_block = f"\nPending checklist steps:\n{lines}\n"
            return (
                "Execute the active flow checklist now.\n"
                f"Original follow-up: {request.strip()}\n"
                f"Current objective: {objective}\n"
                f"{pending_block}"
                "Rules:\n"
                "- Do not return only a new high-level plan.\n"
                "- Start executing pending checklist steps with tool calls.\n"
                "- Apply concrete edits and verification steps when required."
            )
        return (
            "Continue the active coding flow.\n"
            f"Original follow-up: {request.strip()}\n"
            f"Current objective: {objective}\n"
            "Proceed with concrete inspection/edit/verification steps for this objective."
        )

    def _effective_system_prompt_for(self, request: str, *, flow_context: str | None = None) -> str:
        prompt = self.system_prompt
        if self._looks_like_edit_request(request):
            prompt = f"{prompt}\n\n{CODING_AGENT_RECOGNITION_PROMPT}"
        if self.full_auto_mode:
            prompt = f"{prompt}\n\n{FULL_AUTO_EXECUTION_PROMPT}"
        if flow_context:
            prompt = (
                f"{prompt}\n\n{CODING_FLOW_MEMORY_PROMPT}\n\n"
                f"Flow context:\n{flow_context.strip()}"
            )
        return prompt

    @staticmethod
    def _log_worker_event(event: Any) -> None:
        name = ""
        message = ""
        data: dict[str, Any] = {}

        if isinstance(event, dict):
            name = str(event.get("name", "") or "")
            message = str(event.get("message", "") or "")
            maybe_data = event.get("data")
            if isinstance(maybe_data, dict):
                data = maybe_data
        else:
            name = str(getattr(event, "name", "") or "")
            message = str(getattr(event, "message", "") or "")
            maybe_data = getattr(event, "data", {})
            if isinstance(maybe_data, dict):
                data = maybe_data

        message = message.strip()
        if message.startswith("TOOL "):
            logger.info(message)
            return

        tool = str(data.get("tool", "") or "tool")
        if name == "tool_start":
            args = str(data.get("args", "") or "").strip()
            line = f"TOOL start: {tool}"
            if args:
                line += f" | args: {args}"
            logger.info(line)
            return
        if name == "tool_end":
            dt = data.get("duration_seconds")
            if isinstance(dt, (int, float)):
                logger.info(f"TOOL end: {tool} ({float(dt):0.1f}s)")
            else:
                logger.info(f"TOOL end: {tool}")
            return
        if name == "tool_error":
            err = str(data.get("error", "") or "").strip()
            line = f"TOOL error: {tool}"
            if err:
                line += f" - {err}"
            logger.info(line)
            return

    def _plan_checklist_with_source(
        self,
        request: str,
        *,
        flow_context: str | None = None,
    ) -> tuple[FlowChecklist | None, list[str], str]:
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

        try:
            first = self.planner_llm.invoke(messages)
            raw = str(getattr(first, "content", "") or "").strip()
            checklist = self._parse_flow_checklist_json(raw, request=request)
            if len(checklist.steps) > self.plan_max_steps:
                checklist.steps = checklist.steps[: self.plan_max_steps]
            return checklist, warnings, "planner"
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
                checklist = self._parse_flow_checklist_json(repaired_raw, request=request)
                if len(checklist.steps) > self.plan_max_steps:
                    checklist.steps = checklist.steps[: self.plan_max_steps]
                return checklist, warnings, "planner_repair"
            except Exception as exc2:  # pragma: no cover - deterministic fallback
                warnings.append(f"planner repair failed: {exc2}")
                warnings.append("planner fallback: using deterministic checklist")
                fallback = self._fallback_checklist(request)
                if len(fallback.steps) > self.plan_max_steps:
                    fallback.steps = fallback.steps[: self.plan_max_steps]
                return fallback, warnings, "deterministic_fallback"
        except Exception as exc:  # pragma: no cover
            warnings.append(f"planner invocation failed: {exc}")
            warnings.append("planner fallback: using deterministic checklist")
            fallback = self._fallback_checklist(request)
            if len(fallback.steps) > self.plan_max_steps:
                fallback.steps = fallback.steps[: self.plan_max_steps]
            return fallback, warnings, "deterministic_fallback"

    def _plan_checklist(self, request: str, *, flow_context: str | None = None) -> tuple[FlowChecklist | None, list[str]]:
        checklist, warnings, _source = self._plan_checklist_with_source(request, flow_context=flow_context)
        return checklist, warnings

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
        request_for_run = self._rewrite_ambiguous_followup(request, effective_flow_context)
        if request_for_run != request:
            warnings.append("followup_request_rewritten_from_flow_context")

        edit_intent = self._looks_like_edit_request(request_for_run)
        forced_write_retry = False

        # First pass
        answer = call_agent_fn(
            request_text=request_for_run,
            tool_policy=tool_policy,
            flow_context=effective_flow_context,
        )
        changed = sorted(self._git_status_paths().difference(before))
        payload = self._extract_payload(answer) or {}
        trace = payload.get("trace", [])
        if not isinstance(trace, list):
            trace = []

        payload_warnings = payload.get("warnings", [])
        if isinstance(payload_warnings, list):
            warnings.extend(str(item) for item in payload_warnings if str(item).strip())

        trace_rows = [item for item in trace if isinstance(item, dict)]
        combined_trace_rows = list(trace_rows)

        # If patch loop detected and no changes, force a single retry without apply_patch.
        force_fallback, force_reason = self._should_force_write_fallback(
            trace=trace_rows,
            changed_files=changed,
        )
        if force_fallback:
            logger.warning(force_reason)
            warnings.append("forced_write_file_retry_after_patch_noop")
            retry_request = (
                "NOTE: apply_patch is unavailable or failed/no-op repeatedly. "
                "Use write_file fallback (chunked part_index + finalize) with full file content.\n\n"
                f"Original request:\n{request}"
            )
            with self._without_tool("apply_patch"):
                answer = call_agent_fn(
                    request_text=retry_request,
                    tool_policy=tool_policy,
                    flow_context=effective_flow_context,
                )
            forced_write_retry = True
            changed = sorted(self._git_status_paths().difference(before))
            payload = self._extract_payload(answer) or {}
            trace = payload.get("trace", [])
            if not isinstance(trace, list):
                trace = []
            payload_warnings = payload.get("warnings", [])
            if isinstance(payload_warnings, list):
                warnings.extend(str(item) for item in payload_warnings if str(item).strip())
            trace_rows = [item for item in trace if isinstance(item, dict)]
            combined_trace_rows.extend(trace_rows)

        decision, counters, read_count = self._compute_progress(
            checklist=checklist,
            trace=combined_trace_rows,
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
            logger.warning("Forcing write_file fallback: re-running agent with apply_patch disabled.")
            warnings.append("forced_write_file_retry_after_patch_noop")
            retry_request = (
                "NOTE: apply_patch is unavailable or failed/no-op repeatedly. "
                "Use write_file fallback (chunked part_index + finalize) with full file content.\n\n"
                f"Original request:\n{request}"
            )
            with self._without_tool("apply_patch"):
                answer = call_agent_fn(
                    request_text=retry_request,
                    tool_policy=tool_policy,
                    flow_context=effective_flow_context,
                )
            forced_write_retry = True
            changed = sorted(self._git_status_paths().difference(before))
            payload = self._extract_payload(answer) or {}
            trace = payload.get("trace", [])
            if not isinstance(trace, list):
                trace = []
            payload_warnings = payload.get("warnings", [])
            if isinstance(payload_warnings, list):
                warnings.extend(str(item) for item in payload_warnings if str(item).strip())
            trace_rows = [item for item in trace if isinstance(item, dict)]
            combined_trace_rows.extend(trace_rows)
            decision, counters, read_count = self._compute_progress(
                checklist=checklist,
                trace=combined_trace_rows,
                warnings=warnings,
                changed_files=changed,
                required_read_files=required_read_files,
            )

        answer_text = self._extract_answer_text(answer)
        mutation_tools_seen = {str(row.get("tool_name", "")) for row in combined_trace_rows}
        attempted_apply_patch = "apply_patch" in mutation_tools_seen
        attempted_write_file = "write_file" in mutation_tools_seen

        if edit_intent and not changed and attempted_apply_patch:
            warnings.append("mutation_noop_after_apply_patch")
        if edit_intent and not changed and attempted_write_file:
            warnings.append("mutation_noop_after_write_file")

        if (
            edit_intent
            and not changed
            and self._looks_like_conversational_terminal(answer_text)
            and not forced_write_retry
        ):
            warnings.append("edit_intent_conversational_noop_detected")
            retry_request = (
                "Do not ask for confirmation. Execute concrete repository edits now. "
                "If apply_patch fails/no-ops, use write_file full-content fallback and verify changed_files.\n\n"
                f"Original request:\n{request}"
            )
            with self._without_tool("apply_patch"):
                answer = call_agent_fn(
                    request_text=retry_request,
                    tool_policy=tool_policy,
                    flow_context=effective_flow_context,
                )
            changed = sorted(self._git_status_paths().difference(before))
            payload = self._extract_payload(answer) or {}
            trace = payload.get("trace", [])
            if not isinstance(trace, list):
                trace = []
            payload_warnings = payload.get("warnings", [])
            if isinstance(payload_warnings, list):
                warnings.extend(str(item) for item in payload_warnings if str(item).strip())
            trace_rows = [item for item in trace if isinstance(item, dict)]
            combined_trace_rows.extend(trace_rows)
            answer_text = self._extract_answer_text(answer)
            mutation_tools_seen = {str(row.get("tool_name", "")) for row in combined_trace_rows}
            attempted_apply_patch = "apply_patch" in mutation_tools_seen
            attempted_write_file = "write_file" in mutation_tools_seen
            decision, counters, read_count = self._compute_progress(
                checklist=checklist,
                trace=combined_trace_rows,
                warnings=warnings,
                changed_files=changed,
                required_read_files=required_read_files,
            )

        if edit_intent and not changed and attempted_apply_patch and attempted_write_file:
            warnings.append("mutation_exhausted_true_blocker")
            decision = ExecutionDecision(
                phase="blocked",
                tool_call_allowed=False,
                why=(
                    "mutation_exhausted_true_blocker: apply_patch and write_file produced no file changes "
                    "after bounded retries."
                ),
            )

        trace_rows = combined_trace_rows
        findings = self._run_static_analysis([p for p in changed if p.endswith(".py")])
        diff = self._git_diff(changed)
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
        trace_rows = [item for item in trace if isinstance(item, dict)]
        warning_text = "\n".join(str(item) for item in warnings).lower()
        tools_only_fallback = (
            ("tools_only_violation" in warning_text)
            and len(trace_rows) == 0
            and not changed
        )
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
            "actions_taken_total": len(trace_rows),
            "actions_taken_truncated": len(trace_rows) > 20,
            "actions_taken": trace_rows[:20],
            "next_step": checklist.next_action or decision.why,
            "static_analysis": {
                "finding_count": len(findings),
                "findings": [_as_jsonable(f) for f in findings],
            },
            "render_mode": "answer_only" if tools_only_fallback else "default",
            "fallback_reason": "tools_only_violation" if tools_only_fallback else "",
            "fallback_retry_attempted": bool("tools_only_violation_retry_attempted" in warning_text),
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
        def _call(*, request_text: str, tool_policy: dict[str, Any], flow_context: str | None) -> str:
            return self._call_agent_single(
                request_text,
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
        def _call(*, request_text: str, tool_policy: dict[str, Any], flow_context: str | None) -> str:
            return self._call_agent_multi(
                request_text,
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
        if self.tool_worker_client is not None:
            tool_req = ToolRunRequest(
                question=request,
                index_dir=str(Path(index_dir).resolve()) if index_dir is not None else None,
                k=int(k if k is not None else 8),
                max_steps=max_steps,
                timeout_seconds=timeout_seconds,
                tool_policy=tool_policy,
                system_prompt=effective_prompt,
            )
            try:
                try:
                    response = self.tool_worker_client.run_tools(
                        tool_req,
                        on_event=self._log_worker_event,
                    )
                except TypeError:
                    response = self.tool_worker_client.run_tools(tool_req)
                return self._stringify(response.model_dump())
            except ToolWorkerProcessError as exc:
                if exc.code == "tools_only_violation":
                    retry_req = tool_req.model_copy(update={"tools_only_strict_override": False})
                    retry_warnings = [
                        f"tools_only_violation: {exc}",
                        "tools_only_violation_retry_attempted: strict override disabled for one retry",
                    ]
                    try:
                        try:
                            retry_response = self.tool_worker_client.run_tools(
                                retry_req,
                                on_event=self._log_worker_event,
                            )
                        except TypeError:
                            retry_response = self.tool_worker_client.run_tools(retry_req)
                        payload = retry_response.model_dump()
                        existing = [str(item) for item in payload.get("warnings", []) if str(item).strip()]
                        payload["warnings"] = [*existing, *retry_warnings, "tools_only_violation_retry_result: success"]
                        payload["fallback_reason"] = "tools_only_violation"
                        payload["fallback_retry_attempted"] = True
                        return self._stringify(payload)
                    except ToolWorkerProcessError as retry_exc:
                        retry_warnings.append(f"tools_only_violation_retry_failed: {retry_exc}")
                    return self._stringify(
                        {
                            "answer": (
                                "Request blocked by tools-only worker policy: no successful tool calls were made. "
                                "Please provide a tool-executable request with specific files or operations."
                            ),
                            "trace": [],
                            "warnings": retry_warnings,
                            "render_mode": "answer_only",
                            "fallback_reason": "tools_only_violation",
                            "fallback_retry_attempted": True,
                        }
                    )
                raise
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
        if self.tool_worker_client is not None:
            tool_req = ToolRunRequest(
                question=request,
                index_dirs=resolved,
                k=int(k if k is not None else 8),
                max_steps=max_steps,
                timeout_seconds=timeout_seconds,
                tool_policy=tool_policy,
                system_prompt=effective_prompt,
            )
            try:
                try:
                    response = self.tool_worker_client.run_tools(
                        tool_req,
                        on_event=self._log_worker_event,
                    )
                except TypeError:
                    response = self.tool_worker_client.run_tools(tool_req)
                return self._stringify(response.model_dump())
            except ToolWorkerProcessError as exc:
                if exc.code == "tools_only_violation":
                    retry_req = tool_req.model_copy(update={"tools_only_strict_override": False})
                    retry_warnings = [
                        f"tools_only_violation: {exc}",
                        "tools_only_violation_retry_attempted: strict override disabled for one retry",
                    ]
                    try:
                        try:
                            retry_response = self.tool_worker_client.run_tools(
                                retry_req,
                                on_event=self._log_worker_event,
                            )
                        except TypeError:
                            retry_response = self.tool_worker_client.run_tools(retry_req)
                        payload = retry_response.model_dump()
                        existing = [str(item) for item in payload.get("warnings", []) if str(item).strip()]
                        payload["warnings"] = [*existing, *retry_warnings, "tools_only_violation_retry_result: success"]
                        payload["fallback_reason"] = "tools_only_violation"
                        payload["fallback_retry_attempted"] = True
                        return self._stringify(payload)
                    except ToolWorkerProcessError as retry_exc:
                        retry_warnings.append(f"tools_only_violation_retry_failed: {retry_exc}")
                    return self._stringify(
                        {
                            "answer": (
                                "Request blocked by tools-only worker policy: no successful tool calls were made. "
                                "Please provide a tool-executable request with specific files or operations."
                            ),
                            "trace": [],
                            "warnings": retry_warnings,
                            "render_mode": "answer_only",
                            "fallback_reason": "tools_only_violation",
                            "fallback_retry_attempted": True,
                        }
                    )
                raise
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
