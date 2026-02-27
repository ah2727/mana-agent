from __future__ import annotations

from pathlib import Path

from mana_analyzer.llm.tool_worker_process import ToolRunResponse
from mana_analyzer.llm.tools_manager import (
    AutoExecuteResult,
    ToolsManagerBatch,
    ToolsManagerOrchestrator,
    ToolsPlan,
)


class _NoopWorker:
    def run_tools(self, _request, on_event=None):  # noqa: ANN001
        _ = on_event
        return ToolRunResponse(answer="ok", sources=[], mode="agent-tools", trace=[], warnings=[])


def _build_orchestrator(tmp_path: Path) -> ToolsManagerOrchestrator:
    orchestrator = object.__new__(ToolsManagerOrchestrator)
    orchestrator.llm = None
    orchestrator.worker_client = _NoopWorker()
    orchestrator.repo_root = tmp_path
    return orchestrator


def test_tools_manager_planner_schema_parses_strict_json(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)

    valid_plan = (
        '{"objective":"Ship feature","steps":['
        '{"id":"s1","title":"Inspect files","tool_intent":"inspect","args_hint":"read foo.py",'
        '"success_signal":"inspected","fallback":"search"}'
        '],"stop_conditions":["done"],"finalize_action":"summarize"}'
    )
    monkeypatch.setattr(orchestrator, "_invoke_model", lambda **_kwargs: valid_plan)

    plan, warnings = orchestrator._plan(request="implement plan", flow_context=None)
    assert warnings == []
    assert isinstance(plan, ToolsPlan)
    assert plan is not None
    assert plan.objective == "Ship feature"
    assert len(plan.steps) == 1


def test_tools_manager_planner_parser_accepts_markdown_plan_text(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    markdown_plan = (
        "Execution Plan:\\n"
        "1. Inspect src/mana_analyzer/llm/coding_agent.py\\n"
        "2. Apply targeted patch\\n"
        "3. Verify with tests\\n"
    )
    plan = orchestrator.parse_tools_plan(markdown_plan, request="implement planner", previous_plan=None)
    assert isinstance(plan, ToolsPlan)
    assert plan is not None
    assert plan.decision == "continue"
    assert len(plan.steps) >= 3


def test_tools_manager_planner_parser_accepts_wrapped_payload(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    wrapped = (
        "[{'type': 'text', 'text': '{\"objective\": \"Ship feature\", \"steps\": "
        "[{\"id\": \"s1\", \"title\": \"Inspect\", \"tool_intent\": \"inspect\", "
        "\"args_hint\": \"\", \"success_signal\": \"\", \"fallback\": \"\", \"status\": \"in_progress\"}], "
        "\"current_step_id\": \"s1\", \"decision\": \"continue\", \"decision_reason\": \"start\", "
        "\"stop_conditions\": [\"done\"], \"finalize_action\": \"summarize\"}'}]"
    )
    plan = orchestrator.parse_tools_plan(wrapped, request="execute", previous_plan=None)
    assert isinstance(plan, ToolsPlan)
    assert plan is not None
    assert plan.objective == "Ship feature"
    assert plan.current_step_id == "s1"


def test_tools_manager_invalid_batch_triggers_repair_then_terminal_stop(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)

    plan = ToolsPlan(
        objective="Run",
        steps=[],
        stop_conditions=["done"],
        finalize_action="answer",
    )

    responses = iter(["{ broken", "not-json"])
    monkeypatch.setattr(orchestrator, "_invoke_model", lambda **_kwargs: next(responses))

    batch, issues = orchestrator._build_batch(
        request="execute",
        flow_context=None,
        plan=plan,
        pass_index=1,
        pass_cap=4,
        pass_logs=[],
        warnings=[],
        changed_files=[],
    )
    assert batch is None
    assert any("attempting repair" in issue for issue in issues)
    assert any("repair failed" in issue for issue in issues)


def test_tools_manager_planner_invalid_uses_deterministic_fallback(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    monkeypatch.setattr(orchestrator, "_invoke_model", lambda **_kwargs: "not-json-at-all")
    plan, warnings = orchestrator._plan(
        request="implement planner",
        flow_context=None,
        pass_index=0,
        pass_cap=4,
        previous_plan=None,
        pass_logs=[],
        warnings=[],
        changed_files=[],
        latest_answer="",
    )
    assert isinstance(plan, ToolsPlan)
    assert plan.steps
    assert plan.decision == "continue"
    assert any("deterministic fallback" in warning for warning in warnings)


def test_tools_manager_stalled_twice_stops_loop(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)

    plan = ToolsPlan(
        objective="Run",
        steps=[],
        stop_conditions=["done"],
        finalize_action="answer",
    )
    monkeypatch.setattr(orchestrator, "_plan", lambda **_kwargs: (plan, []))
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(requests=[], continue_after=True, expected_progress="waiting"),
            [],
        ),
    )

    result = orchestrator.run(
        request="execute",
        flow_context=None,
        index_dir=tmp_path / ".mana_index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=6,
    )
    assert isinstance(result, AutoExecuteResult)
    assert result.terminal_reason == "stalled_no_actionable_requests"
    assert result.passes == 2


def test_tools_manager_invalid_request_batch_terminal_reason(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)

    plan = ToolsPlan(
        objective="Run",
        steps=[],
        stop_conditions=["done"],
        finalize_action="answer",
    )
    monkeypatch.setattr(orchestrator, "_plan", lambda **_kwargs: (plan, []))
    monkeypatch.setattr(orchestrator, "_build_batch", lambda **_kwargs: (None, ["bad batch"]))

    result = orchestrator.run(
        request="execute",
        flow_context=None,
        index_dir=tmp_path / ".mana_index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=4,
    )
    assert result.terminal_reason == "invalid_request_batch"
    assert "bad batch" in result.warnings
    assert result.passes == 0
