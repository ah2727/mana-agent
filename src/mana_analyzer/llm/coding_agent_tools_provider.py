from __future__ import annotations

import json
from typing import Any, Protocol, Sequence

from mana_analyzer.llm.tools_manager import (
    ToolsManagerBatch,
    ToolsManagerDecisionProvider,
    ToolsPlan,
)


class _CodingAgentPlannerLike(Protocol):
    tools_manager_orchestrator: Any

    def _invoke_tools_planner(self, human_prompt: str) -> str: ...

    def _repair_tools_planner(self, raw: str, human_prompt: str) -> str: ...

    def _invoke_tools_batcher(self, human_prompt: str) -> str: ...

    def _repair_tools_batcher(self, raw: str, human_prompt: str) -> str: ...


class CodingAgentToolsManagerDecisionProvider(ToolsManagerDecisionProvider):
    def __init__(self, agent: _CodingAgentPlannerLike) -> None:
        self.agent = agent

    def plan_with_source(
        self,
        *,
        request: str,
        flow_context: str | None,
        pass_index: int,
        pass_cap: int,
        previous_plan: ToolsPlan | None,
        pass_logs: Sequence[dict[str, Any]],
        warnings: Sequence[str],
        changed_files: Sequence[str],
        latest_answer: str,
    ) -> tuple[ToolsPlan, list[str], str]:
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

        raw = self.agent._invoke_tools_planner(human_prompt)
        parsed = self.agent.tools_manager_orchestrator.parse_tools_plan(
            raw,
            request=request,
            previous_plan=previous_plan,
        )
        if parsed is not None:
            return parsed, [], "planner"

        repair_raw = self.agent._repair_tools_planner(raw, human_prompt)
        repaired = self.agent.tools_manager_orchestrator.parse_repair(
            repair_raw,
            "plan",
            request=request,
            previous_plan=previous_plan,
        )
        if isinstance(repaired, ToolsPlan):
            return repaired, ["head_tools_planner parse failed; attempting repair"], "planner_repair"

        fallback_plan = self.agent.tools_manager_orchestrator._deterministic_fallback_plan(
            request=request,
            flow_context=flow_context,
            previous_plan=previous_plan,
            reason="planner_parse_failed",
        )
        issues = [
            "head_tools_planner parse failed; attempting repair",
            "head_tools_planner repair failed; using deterministic fallback",
        ]
        return fallback_plan, issues, "deterministic_fallback"

    def build_batch(
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
        latest_answer: str,
    ) -> tuple[ToolsManagerBatch | None, list[str]]:
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
        raw = self.agent._invoke_tools_batcher(human_prompt)
        batch = self.agent.tools_manager_orchestrator.parse_tools_batch(
            raw,
            planner_step_id=plan.current_step_id,
        )
        if batch is not None:
            return batch, []

        repair_raw = self.agent._repair_tools_batcher(raw, human_prompt)
        repaired_batch = self.agent.tools_manager_orchestrator.parse_repair(
            repair_raw,
            "batch",
            request=request,
            previous_plan=plan,
            planner_step_id=plan.current_step_id,
        )
        if isinstance(repaired_batch, ToolsManagerBatch):
            return repaired_batch, ["toolsmanager batch invalid; attempting repair"]

        return None, ["toolsmanager batch invalid; attempting repair", "toolsmanager repair failed"]

