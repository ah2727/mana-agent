from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from mana_analyzer.llm.prompts import HEAD_TOOLS_PLANNER_PROMPT, TOOLSMANAGER_PROMPT
from mana_analyzer.llm.tool_worker_process import ToolRunRequest, ToolWorkerClient, ToolWorkerProcessError


PlanDecision = Literal["continue", "revise", "finalize", "stop"]
StepStatus = Literal["pending", "in_progress", "done", "blocked"]
_ModelT = TypeVar("_ModelT", bound=BaseModel)


class ToolsPlanStep(BaseModel):
    id: str
    title: str
    tool_intent: Literal["inspect", "search", "edit", "verify", "answer"]
    args_hint: str = ""
    success_signal: str = ""
    fallback: str = ""
    status: StepStatus = "pending"


class ToolsPlan(BaseModel):
    objective: str
    steps: list[ToolsPlanStep] = Field(default_factory=list)
    current_step_id: str = ""
    decision: PlanDecision = "continue"
    decision_reason: str = ""
    stop_conditions: list[str] = Field(default_factory=list)
    finalize_action: str = ""


class ToolsManagerRequest(BaseModel):
    question: str
    tool_policy_override: dict[str, Any] | None = None
    timeout_seconds: int | None = None


class ToolsManagerBatch(BaseModel):
    planner_step_id: str = ""
    batch_reason: str = ""
    requests: list[ToolsManagerRequest] = Field(default_factory=list)
    continue_after: bool = True
    expected_progress: str = ""


class AutoExecuteResult(BaseModel):
    answer: str = ""
    sources: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    plan: dict[str, Any] | None = None
    passes: int = 0
    terminal_reason: str = ""
    toolsmanager_requests_count: int = 0
    pass_logs: list[dict[str, Any]] = Field(default_factory=list)
    planner_decisions: list[dict[str, Any]] = Field(default_factory=list)
    prechecklist: dict[str, Any] | None = None
    prechecklist_source: str = ""
    prechecklist_warning: str = ""


class ToolsManagerOrchestrator:
    """Planner-driven auto-execution loop for agent-tools chat turns."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        worker_client: ToolWorkerClient,
        repo_root: Path,
        base_url: str | None = None,
    ) -> None:
        llm_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "model": model,
        }
        if base_url:
            llm_kwargs["base_url"] = base_url
        self.llm = ChatOpenAI(**llm_kwargs)
        self.worker_client = worker_client
        self.repo_root = repo_root.resolve()

    @staticmethod
    def _strip_code_fence(raw: str) -> str:
        text = str(raw or "").strip()
        if not text.startswith("```"):
            return text
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
        return text

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
    def _extract_json_object_text(text: str) -> str | None:
        raw = str(text or "").strip()
        if not raw:
            return None
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
                    return raw[start : idx + 1].strip()
        return None

    @classmethod
    def _collect_candidates(cls, raw_text: str) -> list[Any]:
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

    @classmethod
    def _parse_model(cls, text: str, model_cls: type[_ModelT]) -> _ModelT:
        last_error: Exception | None = None
        for candidate in cls._collect_candidates(text):
            if isinstance(candidate, dict):
                try:
                    return model_cls.model_validate(candidate)
                except Exception as exc:
                    last_error = exc
            elif isinstance(candidate, str):
                parsed = cls._parse_json_or_literal(candidate)
                if isinstance(parsed, dict):
                    try:
                        return model_cls.model_validate(parsed)
                    except Exception as exc:
                        last_error = exc
        if last_error is not None:
            raise ValueError(str(last_error)) from last_error
        raise ValueError("No valid JSON object found")

    @staticmethod
    def _intent_from_text(text: str) -> Literal["inspect", "search", "edit", "verify", "answer"]:
        lowered = str(text or "").lower()
        if any(token in lowered for token in ("search", "semantic", "find", "lookup")):
            return "search"
        if any(token in lowered for token in ("edit", "patch", "write", "update", "change", "refactor")):
            return "edit"
        if any(token in lowered for token in ("verify", "test", "lint", "mypy", "ruff", "check")):
            return "verify"
        if any(token in lowered for token in ("answer", "summar", "final", "report")):
            return "answer"
        return "inspect"

    @staticmethod
    def _status_from_text(text: str) -> StepStatus:
        lowered = str(text or "").strip().lower()
        if lowered in {"pending", "in_progress", "done", "blocked"}:
            return lowered  # type: ignore[return-value]
        return "pending"

    def _normalize_plan(self, plan: ToolsPlan, *, previous_plan: ToolsPlan | None = None) -> ToolsPlan:
        steps: list[ToolsPlanStep] = []
        seen_ids: set[str] = set()
        for idx, step in enumerate(plan.steps, start=1):
            base_id = str(step.id or "").strip() or f"s{idx}"
            step_id = base_id
            suffix = 1
            while step_id in seen_ids:
                suffix += 1
                step_id = f"{base_id}_{suffix}"
            seen_ids.add(step_id)
            steps.append(
                ToolsPlanStep(
                    id=step_id,
                    title=str(step.title or "").strip() or f"Step {idx}",
                    tool_intent=step.tool_intent,
                    args_hint=str(step.args_hint or "").strip(),
                    success_signal=str(step.success_signal or "").strip(),
                    fallback=str(step.fallback or "").strip(),
                    status=self._status_from_text(step.status),
                )
            )

        if not steps and previous_plan is not None and previous_plan.steps:
            steps = [ToolsPlanStep.model_validate(item.model_dump()) for item in previous_plan.steps]

        if not steps:
            steps = [
                ToolsPlanStep(
                    id="s1",
                    title="Inspect target files",
                    tool_intent="inspect",
                    args_hint="Use semantic_search/read_file on concrete files.",
                    success_signal="relevant file context gathered",
                    fallback="If unknown files, run targeted search once.",
                    status="in_progress",
                ),
                ToolsPlanStep(
                    id="s2",
                    title="Apply requested changes",
                    tool_intent="edit",
                    args_hint="Use apply_patch first, write_file fallback if needed.",
                    success_signal="requested edits applied",
                    fallback="Use write_file if patch loop fails twice.",
                    status="pending",
                ),
                ToolsPlanStep(
                    id="s3",
                    title="Verify and finalize",
                    tool_intent="verify",
                    args_hint="Run targeted verification and summarize.",
                    success_signal="verification complete",
                    fallback="If verification tooling unavailable, state limits and remaining risk.",
                    status="pending",
                ),
            ]

        objective = str(plan.objective or "").strip() or "Execute requested plan"
        decision: PlanDecision = str(plan.decision or "continue").strip().lower()  # type: ignore[assignment]
        if decision not in {"continue", "revise", "finalize", "stop"}:
            decision = "continue"

        current_step_id = str(plan.current_step_id or "").strip()
        if current_step_id not in {step.id for step in steps}:
            active = next((step for step in steps if step.status not in {"done", "blocked"}), None)
            current_step_id = active.id if active is not None else steps[0].id

        if decision in {"finalize", "stop"} and not str(plan.decision_reason or "").strip():
            decision_reason = "Planner marked terminal decision."
        else:
            decision_reason = str(plan.decision_reason or "").strip()

        stop_conditions = [str(item).strip() for item in plan.stop_conditions if str(item).strip()]
        if not stop_conditions:
            stop_conditions = [
                "Planner chooses finalize/stop",
                "Two consecutive non-actionable passes",
                "Pass cap reached",
            ]

        finalize_action = str(plan.finalize_action or "").strip() or "Return final answer with completed work and verification."

        return ToolsPlan(
            objective=objective,
            steps=steps,
            current_step_id=current_step_id,
            decision=decision,
            decision_reason=decision_reason,
            stop_conditions=stop_conditions,
            finalize_action=finalize_action,
        )

    def _plan_from_markdown_text(
        self,
        text: str,
        *,
        request: str,
        previous_plan: ToolsPlan | None = None,
    ) -> ToolsPlan | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        if "\\n" in raw and "\n" not in raw:
            raw = raw.replace("\\n", "\n")

        objective = ""
        extracted_steps: list[dict[str, str]] = []

        for line in raw.splitlines():
            item = line.strip()
            if not item:
                continue
            lowered = item.lower()
            if lowered.startswith("objective:"):
                objective = item.split(":", 1)[1].strip()
                continue
            if lowered in {"plan:", "execution plan:", "**plan**", "**execution plan**"}:
                continue

            match = re.match(r"^(?:\d+[.)]\s+|[-*]\s+)(.+)$", item)
            if not match:
                continue
            payload = match.group(1).strip()
            status = "pending"
            status_match = re.match(r"^\[(pending|in_progress|done|blocked)\]\s+(.+)$", payload, flags=re.IGNORECASE)
            if status_match:
                status = status_match.group(1).lower()
                payload = status_match.group(2).strip()
            if payload:
                extracted_steps.append({"title": payload[:220], "status": status})

        if not extracted_steps and previous_plan is not None and previous_plan.steps:
            extracted_steps = [
                {
                    "title": str(item.title),
                    "status": str(item.status),
                }
                for item in previous_plan.steps
            ]

        if not extracted_steps:
            return None

        steps: list[ToolsPlanStep] = []
        for idx, item in enumerate(extracted_steps, start=1):
            title = str(item.get("title", "") or f"Step {idx}").strip()
            status = self._status_from_text(item.get("status", "pending"))
            if idx == 1 and status == "pending":
                status = "in_progress"
            steps.append(
                ToolsPlanStep(
                    id=f"s{idx}",
                    title=title,
                    tool_intent=self._intent_from_text(title),
                    args_hint="Derived from planner markdown fallback",
                    success_signal="",
                    fallback="Use repository evidence and continue safely.",
                    status=status,
                )
            )

        resolved_objective = objective or (" ".join((request or "").strip().split())[:220] or "Execute requested plan")
        return self._normalize_plan(
            ToolsPlan(
                objective=resolved_objective,
                steps=steps,
                decision="continue",
                decision_reason="Recovered plan from markdown/text output.",
                stop_conditions=[],
                finalize_action="Return final answer with completed work.",
            ),
            previous_plan=previous_plan,
        )

    def _deterministic_fallback_plan(
        self,
        *,
        request: str,
        flow_context: str | None,
        previous_plan: ToolsPlan | None,
        reason: str,
    ) -> ToolsPlan:
        if previous_plan is not None:
            base = self._normalize_plan(previous_plan, previous_plan=previous_plan)
            return ToolsPlan(
                objective=base.objective,
                steps=base.steps,
                current_step_id=base.current_step_id,
                decision="continue",
                decision_reason=f"Deterministic fallback: {reason}",
                stop_conditions=base.stop_conditions,
                finalize_action=base.finalize_action,
            )

        context_hint = ""
        if flow_context:
            for line in str(flow_context).splitlines():
                text = line.strip()
                if text.lower().startswith("current objective:"):
                    context_hint = text.split(":", 1)[1].strip()
                    break

        objective = context_hint or (" ".join((request or "").strip().split())[:220] or "Execute requested plan")
        plan = ToolsPlan(
            objective=objective,
            steps=[],
            current_step_id="",
            decision="continue",
            decision_reason=f"Deterministic fallback: {reason}",
            stop_conditions=[],
            finalize_action="Return final answer with completed work.",
        )
        return self._normalize_plan(plan, previous_plan=None)

    def parse_tools_plan(
        self,
        raw_text: str,
        *,
        request: str,
        previous_plan: ToolsPlan | None = None,
    ) -> ToolsPlan | None:
        try:
            parsed = self._parse_model(raw_text, ToolsPlan)
            return self._normalize_plan(parsed, previous_plan=previous_plan)
        except Exception:
            pass
        direct = self._plan_from_markdown_text(raw_text, request=request, previous_plan=previous_plan)
        if direct is not None:
            return direct
        for candidate in self._collect_candidates(raw_text):
            if not isinstance(candidate, str):
                continue
            parsed_text = self._plan_from_markdown_text(candidate, request=request, previous_plan=previous_plan)
            if parsed_text is not None:
                return parsed_text
        return None

    def _normalize_batch(self, batch: ToolsManagerBatch, *, planner_step_id: str) -> ToolsManagerBatch:
        requests: list[ToolsManagerRequest] = []
        for item in batch.requests[:8]:
            question = str(item.question or "").strip()
            if not question:
                continue
            override = item.tool_policy_override if isinstance(item.tool_policy_override, dict) else None
            timeout = item.timeout_seconds if isinstance(item.timeout_seconds, int) else None
            requests.append(
                ToolsManagerRequest(
                    question=question,
                    tool_policy_override=override,
                    timeout_seconds=timeout,
                )
            )

        resolved_step = str(batch.planner_step_id or "").strip() or planner_step_id
        return ToolsManagerBatch(
            planner_step_id=resolved_step,
            batch_reason=str(batch.batch_reason or "").strip() or "toolsmanager_batch",
            requests=requests,
            continue_after=bool(batch.continue_after),
            expected_progress=str(batch.expected_progress or "").strip(),
        )

    def parse_tools_batch(
        self,
        raw_text: str,
        *,
        planner_step_id: str,
    ) -> ToolsManagerBatch | None:
        try:
            parsed = self._parse_model(raw_text, ToolsManagerBatch)
            batch = self._normalize_batch(parsed, planner_step_id=planner_step_id)
            self._validate_batch(batch)
            return batch
        except Exception:
            return None

    def parse_repair(
        self,
        raw_text: str,
        schema_kind: Literal["plan", "batch"],
        *,
        request: str,
        previous_plan: ToolsPlan | None = None,
        planner_step_id: str = "",
    ) -> ToolsPlan | ToolsManagerBatch | None:
        if schema_kind == "plan":
            return self.parse_tools_plan(raw_text, request=request, previous_plan=previous_plan)
        return self.parse_tools_batch(raw_text, planner_step_id=planner_step_id)

    @staticmethod
    def _validate_batch(batch: ToolsManagerBatch) -> None:
        for idx, req in enumerate(batch.requests):
            if not str(req.question or "").strip():
                raise ValueError(f"request[{idx}] question must not be empty")

    def _invoke_model(self, *, system_prompt: str, human_prompt: str) -> str:
        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )
        return str(getattr(response, "content", "") or "").strip()

    def _plan(
        self,
        *,
        request: str,
        flow_context: str | None,
        pass_index: int = 0,
        pass_cap: int = 4,
        previous_plan: ToolsPlan | None = None,
        pass_logs: Sequence[dict[str, Any]] = (),
        warnings: Sequence[str] = (),
        changed_files: Sequence[str] = (),
        latest_answer: str = "",
    ) -> tuple[ToolsPlan, list[str]]:
        plan, issues, _source = self._plan_with_source(
            request=request,
            flow_context=flow_context,
            pass_index=pass_index,
            pass_cap=pass_cap,
            previous_plan=previous_plan,
            pass_logs=pass_logs,
            warnings=warnings,
            changed_files=changed_files,
            latest_answer=latest_answer,
        )
        return plan, issues

    def _build_batch(
        self,
        *,
        request: str,
        flow_context: str | None,
        plan: ToolsPlan,
        pass_index: int,
        pass_cap: int,
        pass_logs: Sequence[dict[str, Any]],
        warnings: Sequence[str],
        changed_files: Sequence[str],
        latest_answer: str = "",
    ) -> tuple[ToolsManagerBatch | None, list[str]]:
        issues: list[str] = []
        payload = {
            "request": request,
            "flow_context": (flow_context or "").strip(),
            "planner": plan.model_dump(),
            "pass_index": int(pass_index),
            "pass_cap": int(pass_cap),
            "pass_logs": list(pass_logs)[-4:],
            "warnings": list(warnings)[-10:],
            "changed_files": list(changed_files),
            "latest_answer": str(latest_answer or "")[:1500],
        }
        human_prompt = json.dumps(payload, ensure_ascii=False, indent=2)

        raw = self._invoke_model(system_prompt=TOOLSMANAGER_PROMPT, human_prompt=human_prompt)
        batch = self.parse_tools_batch(raw, planner_step_id=plan.current_step_id)
        if batch is not None:
            return batch, issues

        issues.append("toolsmanager batch invalid; attempting repair")
        repaired = self._invoke_model(
            system_prompt=TOOLSMANAGER_PROMPT,
            human_prompt=(
                "Repair this tools-manager output to strict JSON schema.\n"
                "Do not add markdown. Return only one JSON object.\n\n"
                f"Broken output:\n{raw}\n\n"
                f"Execution payload:\n{human_prompt}"
            ),
        )
        repaired_batch = self.parse_repair(
            repaired,
            "batch",
            request=request,
            previous_plan=plan,
            planner_step_id=plan.current_step_id,
        )
        if isinstance(repaired_batch, ToolsManagerBatch):
            return repaired_batch, issues

        issues.append("toolsmanager repair failed")
        return None, issues

    @staticmethod
    def _merge_policy(base_policy: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
        merged = dict(base_policy)
        if isinstance(override, dict):
            for key, value in override.items():
                merged[key] = value
        return merged

    @staticmethod
    def _clip_timeout(value: int | None, *, session_timeout: int) -> int:
        base = int(value or session_timeout)
        return max(5, min(base, max(5, int(session_timeout))))

    @staticmethod
    def _fingerprint_request(question: str, policy: dict[str, Any], timeout_seconds: int) -> str:
        raw = json.dumps(
            {
                "question": question,
                "policy": policy,
                "timeout_seconds": timeout_seconds,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

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
                    path = line[3:].strip()
                    if path:
                        paths.add(path.replace("\\", "/"))
            return paths
        except Exception:
            return set()

    def _resolve_step(self, plan: ToolsPlan) -> ToolsPlanStep | None:
        for step in plan.steps:
            if step.id == plan.current_step_id:
                return step
        return plan.steps[0] if plan.steps else None

    @staticmethod
    def _planner_decision_row(plan: ToolsPlan, pass_index: int) -> dict[str, Any]:
        step = next((item for item in plan.steps if item.id == plan.current_step_id), None)
        return {
            "pass_index": int(pass_index),
            "current_step_id": str(plan.current_step_id or ""),
            "current_step_title": str(getattr(step, "title", "") or ""),
            "decision": str(plan.decision or "continue"),
            "decision_reason": str(plan.decision_reason or ""),
        }

    @staticmethod
    def _truncate_line(value: str, *, limit: int = 220) -> str:
        text = " ".join(str(value or "").strip().split())
        if len(text) <= limit:
            return text
        return text[: limit - 1].rstrip() + "…"

    def _deterministic_fallback_request(
        self,
        *,
        request: str,
        flow_context: str | None,
        plan: ToolsPlan,
        step: ToolsPlanStep | None,
        pass_index: int,
    ) -> ToolsManagerRequest | None:
        if step is None:
            return None

        step_title = self._truncate_line(step.title or "current step")
        step_hint = self._truncate_line(step.args_hint or step.fallback or step.success_signal or "")
        objective = self._truncate_line(plan.objective or request or "Execute requested plan")
        flow = self._truncate_line(flow_context or "", limit=320)
        user_request = self._truncate_line(request or "", limit=320)

        if step.tool_intent == "inspect":
            directive = (
                "Inspect repository files for this step using semantic_search/read_file/run_command. "
                "Gather concrete evidence with file paths and line ranges."
            )
        elif step.tool_intent == "search":
            directive = (
                "Run targeted repository search for the requested behavior and gather concrete file evidence "
                "before proposing edits."
            )
        elif step.tool_intent == "edit":
            directive = (
                "Apply concrete repository edits for this step. Prefer apply_patch first and use write_file fallback "
                "if patch application fails."
            )
        elif step.tool_intent == "verify":
            directive = (
                "Verify relevant changes with targeted checks (tests/lint/type checks or focused run_command checks), "
                "then summarize verification evidence."
            )
        else:
            directive = (
                "Summarize current status with concrete repository evidence and identify the next actionable step."
            )

        lines = [
            f"Deterministic fallback request for planner pass {int(pass_index)}.",
            f"Objective: {objective}",
            f"Planner step: {step_title}",
            f"Intent: {step.tool_intent}",
            f"Original request: {user_request or '-'}",
        ]
        if step_hint:
            lines.append(f"Step hint: {step_hint}")
        if flow:
            lines.append(f"Flow context: {flow}")
        lines.append(f"Action: {directive}")
        return ToolsManagerRequest(question="\n".join(lines))

    @staticmethod
    def _synthesize_terminal_answer(
        *,
        terminal_reason: str,
        pass_logs: Sequence[dict[str, Any]],
        planner_decisions: Sequence[dict[str, Any]],
        toolsmanager_requests_count: int,
    ) -> str:
        reason = str(terminal_reason or "unknown").strip() or "unknown"
        passes = len(pass_logs)
        last_pass = pass_logs[-1] if pass_logs else {}
        if not isinstance(last_pass, dict):
            last_pass = {}
        last_step = str(last_pass.get("planner_step_title", "") or "").strip()
        decision_reason = ""
        if planner_decisions:
            tail = planner_decisions[-1]
            if isinstance(tail, dict):
                decision_reason = str(tail.get("decision_reason", "") or "").strip()
        if not decision_reason:
            decision_reason = str(last_pass.get("planner_decision_reason", "") or "").strip()

        lines = [
            "Auto-execute ended without a direct answer from tool runs.",
            f"terminal_reason={reason}",
            f"passes={passes}",
            f"toolsmanager_requests={int(toolsmanager_requests_count)}",
        ]
        if last_step:
            lines.append(f"last_step={last_step}")
        if decision_reason:
            lines.append(f"planner_reason={decision_reason}")
        return "\n".join(lines)

    @staticmethod
    def _normalize_prechecklist(plan: ToolsPlan, *, source: str) -> dict[str, Any]:
        steps: list[dict[str, str]] = []
        for item in plan.steps[:20]:
            steps.append(
                {
                    "id": str(item.id or "").strip() or "step",
                    "title": str(item.title or "").strip() or "step",
                    "status": str(item.status or "pending"),
                }
            )
        return {
            "objective": str(plan.objective or "").strip(),
            "steps": steps,
            "source": str(source or ""),
        }

    def preview_plan(
        self,
        *,
        request: str,
        flow_context: str | None,
        pass_cap: int,
    ) -> dict[str, Any]:
        plan, warnings, source = self._plan_with_source(
            request=request,
            flow_context=flow_context,
            pass_index=0,
            pass_cap=pass_cap,
            previous_plan=None,
            pass_logs=[],
            warnings=[],
            changed_files=[],
            latest_answer="",
        )
        warning_text = ""
        if source == "deterministic_fallback":
            warning_text = "Planner parse failed; using deterministic fallback checklist."
        return {
            "prechecklist": self._normalize_prechecklist(plan, source=source),
            "prechecklist_source": source,
            "prechecklist_warning": warning_text,
            "warnings": warnings,
        }

    def _plan_with_source(
        self,
        *,
        request: str,
        flow_context: str | None,
        pass_index: int = 0,
        pass_cap: int = 4,
        previous_plan: ToolsPlan | None = None,
        pass_logs: Sequence[dict[str, Any]] = (),
        warnings: Sequence[str] = (),
        changed_files: Sequence[str] = (),
        latest_answer: str = "",
    ) -> tuple[ToolsPlan, list[str], str]:
        issues: list[str] = []
        payload = {
            "request": request,
            "flow_context": (flow_context or "none").strip(),
            "pass_index": int(pass_index),
            "pass_cap": int(pass_cap),
            "previous_plan": previous_plan.model_dump() if previous_plan is not None else None,
            "pass_logs": list(pass_logs)[-4:],
            "warnings": list(warnings)[-12:],
            "changed_files": list(changed_files),
            "latest_answer": str(latest_answer or "")[:1500],
        }
        human_prompt = json.dumps(payload, ensure_ascii=False, indent=2)

        raw = self._invoke_model(system_prompt=HEAD_TOOLS_PLANNER_PROMPT, human_prompt=human_prompt)
        parsed = self.parse_tools_plan(raw, request=request, previous_plan=previous_plan)
        if parsed is not None:
            return parsed, issues, "planner"

        issues.append("head_tools_planner parse failed; attempting repair")
        repaired_raw = self._invoke_model(
            system_prompt=HEAD_TOOLS_PLANNER_PROMPT,
            human_prompt=(
                "Repair this planner output to strict JSON schema.\n"
                "Do not add markdown. Return only one JSON object.\n\n"
                f"Broken output:\n{raw}\n\n"
                f"Execution payload:\n{human_prompt}"
            ),
        )
        repaired = self.parse_repair(
            repaired_raw,
            "plan",
            request=request,
            previous_plan=previous_plan,
        )
        if isinstance(repaired, ToolsPlan):
            return repaired, issues, "planner_repair"

        issues.append("head_tools_planner repair failed; using deterministic fallback")
        fallback_plan = self._deterministic_fallback_plan(
            request=request,
            flow_context=flow_context,
            previous_plan=previous_plan,
            reason="planner_parse_failed",
        )
        return fallback_plan, issues, "deterministic_fallback"

    def run(
        self,
        *,
        request: str,
        flow_context: str | None,
        index_dir: str | Path | None,
        index_dirs: Sequence[str | Path] | None,
        k: int,
        max_steps: int,
        timeout_seconds: int,
        tool_policy: dict[str, Any],
        pass_cap: int,
        on_event: Callable[[Any], None] | None = None,
    ) -> AutoExecuteResult:
        pass_cap = max(1, min(int(pass_cap), 12))
        all_warnings: list[str] = []
        all_trace: list[dict[str, Any]] = []
        all_sources: list[dict[str, Any]] = []
        all_pass_logs: list[dict[str, Any]] = []
        planner_decisions: list[dict[str, Any]] = []
        terminal_reason = "pass_cap_reached"
        toolsmanager_requests_count = 0
        latest_answer = ""
        stalled_passes = 0

        plan, plan_warnings, _source = self._plan_with_source(
            request=request,
            flow_context=flow_context,
            pass_index=0,
            pass_cap=pass_cap,
            previous_plan=None,
            pass_logs=[],
            warnings=[],
            changed_files=[],
            latest_answer="",
        )
        all_warnings.extend(plan_warnings)

        before = self._git_status_paths()
        changed_files: list[str] = []

        for pass_index in range(1, pass_cap + 1):
            step = self._resolve_step(plan)
            planner_row = self._planner_decision_row(plan, pass_index)
            planner_decisions.append(planner_row)

            if plan.decision in {"finalize", "stop"}:
                terminal_reason = "planner_finalize" if plan.decision == "finalize" else "planner_stop"
                if not latest_answer:
                    latest_answer = str(plan.finalize_action or "").strip()
                all_pass_logs.append(
                    {
                        "pass_index": pass_index,
                        "planner_step_id": plan.current_step_id,
                        "planner_step_title": str(getattr(step, "title", "") or ""),
                        "planner_decision": plan.decision,
                        "planner_decision_reason": plan.decision_reason,
                        "batch_reason": "planner_terminal",
                        "expected_progress": "",
                        "requests_count": 0,
                        "request_fingerprints": [],
                        "tool_steps": 0,
                        "warnings_delta": 0,
                    }
                )
                break

            batch, batch_warnings = self._build_batch(
                request=request,
                flow_context=flow_context,
                plan=plan,
                pass_index=pass_index,
                pass_cap=pass_cap,
                pass_logs=all_pass_logs,
                warnings=all_warnings,
                changed_files=changed_files,
                latest_answer=latest_answer,
            )
            all_warnings.extend(batch_warnings)

            if batch is None:
                terminal_reason = "invalid_request_batch"
                break

            if plan.decision == "continue" and not batch.requests:
                fallback_request = self._deterministic_fallback_request(
                    request=request,
                    flow_context=flow_context,
                    plan=plan,
                    step=step,
                    pass_index=pass_index,
                )
                if fallback_request is not None:
                    batch = ToolsManagerBatch(
                        planner_step_id=str(batch.planner_step_id or plan.current_step_id or ""),
                        batch_reason="deterministic_empty_batch_fallback",
                        requests=[fallback_request],
                        continue_after=bool(batch.continue_after),
                        expected_progress=(
                            str(batch.expected_progress or "").strip()
                            or "Execute deterministic fallback request for current planner step."
                        ),
                    )
                    all_warnings.append(
                        f"toolsmanager emitted empty request batch on pass {int(pass_index)}; using deterministic fallback request"
                    )
                else:
                    all_warnings.append(
                        f"toolsmanager emitted empty request batch on pass {int(pass_index)} and no deterministic fallback could be derived"
                    )

            request_fingerprints: list[str] = []
            tool_steps_this_pass = 0
            warnings_before = len(all_warnings)
            executed_requests = 0

            for item in batch.requests:
                merged_policy = self._merge_policy(tool_policy, item.tool_policy_override)
                clipped_timeout = self._clip_timeout(item.timeout_seconds, session_timeout=timeout_seconds)
                question = str(item.question or "").strip()
                if not question:
                    continue
                request_fingerprints.append(self._fingerprint_request(question, merged_policy, clipped_timeout))
                toolsmanager_requests_count += 1
                tool_request = ToolRunRequest(
                    question=question,
                    index_dir=str(Path(index_dir).resolve()) if index_dir is not None else None,
                    index_dirs=[str(Path(p).resolve()) for p in (index_dirs or []) if str(p).strip()] or None,
                    k=int(k),
                    max_steps=int(max_steps),
                    timeout_seconds=clipped_timeout,
                    tool_policy=merged_policy,
                    system_prompt=None,
                )
                try:
                    response = self.worker_client.run_tools(tool_request, on_event=on_event)
                except ToolWorkerProcessError as exc:
                    all_warnings.append(f"toolsmanager worker error: {exc.code}: {exc}")
                    continue

                executed_requests += 1
                if response.answer:
                    latest_answer = str(response.answer)
                if isinstance(response.warnings, list):
                    for warning in response.warnings:
                        text = str(warning).strip()
                        if text:
                            all_warnings.append(text)
                if isinstance(response.trace, list):
                    rows = [row for row in response.trace if isinstance(row, dict)]
                    all_trace.extend(rows)
                    tool_steps_this_pass += len(rows)
                if isinstance(response.sources, list):
                    all_sources.extend([row for row in response.sources if isinstance(row, dict)])

            changed_now = sorted(self._git_status_paths().difference(before))
            changed_files = changed_now
            warnings_delta = max(0, len(all_warnings) - warnings_before)
            all_pass_logs.append(
                {
                    "pass_index": pass_index,
                    "planner_step_id": plan.current_step_id,
                    "planner_step_title": str(getattr(step, "title", "") or ""),
                    "planner_decision": plan.decision,
                    "planner_decision_reason": plan.decision_reason,
                    "batch_reason": str(batch.batch_reason or ""),
                    "expected_progress": str(batch.expected_progress or ""),
                    "requests_count": len(batch.requests),
                    "request_fingerprints": request_fingerprints,
                    "tool_steps": tool_steps_this_pass,
                    "warnings_delta": warnings_delta,
                    "continue_after": bool(batch.continue_after),
                }
            )

            if executed_requests == 0:
                stalled_passes += 1
            else:
                stalled_passes = 0

            if stalled_passes >= 2:
                terminal_reason = "stalled_no_actionable_requests"
                break

            if pass_index >= pass_cap:
                terminal_reason = "pass_cap_reached"
                break

            plan, new_plan_warnings, _source = self._plan_with_source(
                request=request,
                flow_context=flow_context,
                pass_index=pass_index,
                pass_cap=pass_cap,
                previous_plan=plan,
                pass_logs=all_pass_logs,
                warnings=all_warnings,
                changed_files=changed_files,
                latest_answer=latest_answer,
            )
            all_warnings.extend(new_plan_warnings)

        if not str(latest_answer or "").strip():
            latest_answer = self._synthesize_terminal_answer(
                terminal_reason=terminal_reason,
                pass_logs=all_pass_logs,
                planner_decisions=planner_decisions,
                toolsmanager_requests_count=toolsmanager_requests_count,
            )

        return AutoExecuteResult(
            answer=latest_answer,
            sources=all_sources,
            trace=all_trace,
            warnings=all_warnings,
            changed_files=changed_files,
            plan=plan.model_dump(),
            passes=len(all_pass_logs),
            terminal_reason=terminal_reason,
            toolsmanager_requests_count=toolsmanager_requests_count,
            pass_logs=all_pass_logs,
            planner_decisions=planner_decisions,
        )


__all__ = [
    "ToolsPlan",
    "ToolsPlanStep",
    "ToolsManagerRequest",
    "ToolsManagerBatch",
    "AutoExecuteResult",
    "ToolsManagerOrchestrator",
]
