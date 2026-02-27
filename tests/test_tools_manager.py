from __future__ import annotations

from pathlib import Path

from mana_analyzer.llm.tool_worker_process import ToolRunResponse
from mana_analyzer.llm.tools_manager import (
    AutoExecuteResult,
    ToolsManagerBatch,
    ToolsManagerOrchestrator,
    ToolsPlan,
)
from mana_analyzer.llm.tools_executor import BatchExecutionResult


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
    monkeypatch.setattr(orchestrator, "_plan_with_source", lambda **_kwargs: (plan, [], "planner"))
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
    assert result.answer
    assert "terminal_reason=stalled_no_actionable_requests" in result.answer


def test_tools_manager_empty_batch_uses_deterministic_request_fallback(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)

    plan = ToolsPlan(
        objective="Run",
        steps=[
            {
                "id": "s1",
                "title": "Inspect key files",
                "tool_intent": "inspect",
                "args_hint": "read TODO.md and cli.py",
                "status": "in_progress",
            }
        ],
        current_step_id="s1",
        stop_conditions=["done"],
        finalize_action="answer",
    )
    monkeypatch.setattr(orchestrator, "_plan_with_source", lambda **_kwargs: (plan, [], "planner"))
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
        pass_cap=1,
    )
    assert isinstance(result, AutoExecuteResult)
    assert result.terminal_reason == "pass_cap_reached"
    assert result.toolsmanager_requests_count == 1
    assert result.pass_logs
    assert result.pass_logs[0]["requests_count"] == 1
    assert result.pass_logs[0]["batch_reason"] == "deterministic_empty_batch_fallback"
    assert any("deterministic fallback request" in str(item).lower() for item in result.warnings)


def test_tools_manager_invalid_request_batch_terminal_reason(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)

    plan = ToolsPlan(
        objective="Run",
        steps=[],
        stop_conditions=["done"],
        finalize_action="answer",
    )
    monkeypatch.setattr(orchestrator, "_plan_with_source", lambda **_kwargs: (plan, [], "planner"))
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
    assert result.answer
    assert "terminal_reason=invalid_request_batch" in result.answer


def test_tools_manager_preview_plan_returns_normalized_prechecklist_without_worker_calls(
    tmp_path: Path, monkeypatch
) -> None:
    class _BoomWorker:
        def run_tools(self, _request, on_event=None):  # noqa: ANN001
            _ = on_event
            raise AssertionError("preview_plan must not execute worker tools")

    orchestrator = _build_orchestrator(tmp_path)
    orchestrator.worker_client = _BoomWorker()
    plan = ToolsPlan(
        objective="Implement planner flow",
        steps=[],
        stop_conditions=["done"],
        finalize_action="answer",
    )
    monkeypatch.setattr(
        orchestrator,
        "_plan_with_source",
        lambda **_kwargs: (plan, [], "planner"),
    )
    preview = orchestrator.preview_plan(request="implement plan.", flow_context=None, pass_cap=4)
    assert isinstance(preview.get("prechecklist"), dict)
    assert preview.get("prechecklist_source") == "planner"
    assert str(preview.get("prechecklist_warning", "")).strip() == ""


def test_tools_manager_preview_plan_surfaces_deterministic_fallback_warning(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    plan = ToolsPlan(
        objective="Fallback objective",
        steps=[],
        stop_conditions=["done"],
        finalize_action="answer",
    )
    monkeypatch.setattr(
        orchestrator,
        "_plan_with_source",
        lambda **_kwargs: (plan, ["planner parse failed"], "deterministic_fallback"),
    )
    preview = orchestrator.preview_plan(request="implement plan.", flow_context=None, pass_cap=4)
    assert preview.get("prechecklist_source") == "deterministic_fallback"
    assert "deterministic fallback checklist" in str(preview.get("prechecklist_warning", "")).lower()


def test_tools_manager_retries_conversational_finalize_without_edits(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)

    first_plan = ToolsPlan(
        objective="Run",
        steps=[],
        decision="finalize",
        decision_reason="If you want, I can continue with edits.",
        stop_conditions=["done"],
        finalize_action="Reply yes to continue.",
    )
    second_plan = ToolsPlan(
        objective="Run",
        steps=[],
        decision="finalize",
        decision_reason="Completed.",
        stop_conditions=["done"],
        finalize_action="Done.",
    )
    calls = {"count": 0}

    def _plan_with_source(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return first_plan, [], "planner"
        return second_plan, [], "planner"

    monkeypatch.setattr(orchestrator, "_plan_with_source", _plan_with_source)
    monkeypatch.setattr(
        orchestrator,
        "_deterministic_fallback_plan",
        lambda **_kwargs: ToolsPlan(
            objective="Run",
            steps=[
                {
                    "id": "s1",
                    "title": "Inspect files",
                    "tool_intent": "inspect",
                    "args_hint": "read_file",
                    "success_signal": "context gathered",
                    "fallback": "search",
                    "status": "in_progress",
                }
            ],
            current_step_id="s1",
            decision="continue",
            decision_reason="forced retry",
            stop_conditions=["done"],
            finalize_action="done",
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(
                planner_step_id="s1",
                batch_reason="retry",
                requests=[{"question": "Inspect relevant files and continue execution."}],
                continue_after=True,
                expected_progress="retry progress",
            ),
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
        pass_cap=2,
    )
    assert result.toolsmanager_requests_count == 1
    assert any("planner_finalize_conversational_without_edits" in str(item) for item in result.warnings)


def test_tools_manager_retries_non_hard_blocker_terminal_without_edits(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)

    first_plan = ToolsPlan(
        objective="Run",
        steps=[],
        decision="finalize",
        decision_reason="Blocker: I need a scope choice. Please choose option 1 or 2.",
        stop_conditions=["done"],
        finalize_action="Awaiting scope decision.",
    )
    second_plan = ToolsPlan(
        objective="Run",
        steps=[],
        decision="finalize",
        decision_reason="Completed.",
        stop_conditions=["done"],
        finalize_action="Done.",
    )
    calls = {"count": 0}

    def _plan_with_source(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return first_plan, [], "planner"
        return second_plan, [], "planner"

    monkeypatch.setattr(orchestrator, "_plan_with_source", _plan_with_source)
    monkeypatch.setattr(
        orchestrator,
        "_deterministic_fallback_plan",
        lambda **_kwargs: ToolsPlan(
            objective="Run",
            steps=[
                {
                    "id": "s1",
                    "title": "Inspect files",
                    "tool_intent": "inspect",
                    "args_hint": "read_file",
                    "success_signal": "context gathered",
                    "fallback": "search",
                    "status": "in_progress",
                }
            ],
            current_step_id="s1",
            decision="continue",
            decision_reason="forced retry",
            stop_conditions=["done"],
            finalize_action="done",
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(
                planner_step_id="s1",
                batch_reason="retry",
                requests=[{"question": "Inspect relevant files and continue execution."}],
                continue_after=True,
                expected_progress="retry progress",
            ),
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
        pass_cap=2,
    )
    assert result.toolsmanager_requests_count == 1
    assert any("planner_terminal_nonhard_blocker_retry" in str(item) for item in result.warnings)


def test_tools_manager_retries_repository_access_soft_blocker_terminal(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)

    first_plan = ToolsPlan(
        objective="Run",
        steps=[],
        decision="finalize",
        decision_reason=(
            "I'm blocked on making a safe, accurate update because I need to read the current repository files first."
        ),
        stop_conditions=["done"],
        finalize_action="Please share permission to proceed.",
    )
    second_plan = ToolsPlan(
        objective="Run",
        steps=[],
        decision="finalize",
        decision_reason="Completed.",
        stop_conditions=["done"],
        finalize_action="Done.",
    )
    calls = {"count": 0}

    def _plan_with_source(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return first_plan, [], "planner"
        return second_plan, [], "planner"

    monkeypatch.setattr(orchestrator, "_plan_with_source", _plan_with_source)
    monkeypatch.setattr(
        orchestrator,
        "_deterministic_fallback_plan",
        lambda **_kwargs: ToolsPlan(
            objective="Run",
            steps=[
                {
                    "id": "s1",
                    "title": "Inspect files",
                    "tool_intent": "inspect",
                    "args_hint": "read_file",
                    "success_signal": "context gathered",
                    "fallback": "search",
                    "status": "in_progress",
                }
            ],
            current_step_id="s1",
            decision="continue",
            decision_reason="forced retry",
            stop_conditions=["done"],
            finalize_action="done",
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(
                planner_step_id="s1",
                batch_reason="retry",
                requests=[{"question": "Inspect relevant files and continue execution."}],
                continue_after=True,
                expected_progress="retry progress",
            ),
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
        pass_cap=2,
    )
    assert result.toolsmanager_requests_count == 1
    assert any("planner_terminal_nonhard_blocker_retry" in str(item) for item in result.warnings)


def test_tools_manager_preserves_stop_for_hard_blocker(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)

    stop_plan = ToolsPlan(
        objective="Run",
        steps=[],
        decision="stop",
        decision_reason="Permission denied and missing credential for write access.",
        stop_conditions=["done"],
        finalize_action="Blocked.",
    )
    monkeypatch.setattr(orchestrator, "_plan_with_source", lambda **_kwargs: (stop_plan, [], "planner"))

    result = orchestrator.run(
        request="execute",
        flow_context=None,
        index_dir=tmp_path / ".mana_index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=3,
    )
    assert result.terminal_reason == "planner_stop"
    assert result.toolsmanager_requests_count == 0
    assert any("planner_terminal_hard_blocker_stop" in str(item) for item in result.warnings)


def test_tools_manager_merges_batch_results_in_input_order(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)

    continue_plan = ToolsPlan(
        objective="Run",
        steps=[
            {
                "id": "s1",
                "title": "Inspect",
                "tool_intent": "inspect",
                "args_hint": "",
                "success_signal": "",
                "fallback": "",
                "status": "in_progress",
            }
        ],
        current_step_id="s1",
        decision="continue",
        stop_conditions=["done"],
        finalize_action="done",
    )
    finalize_plan = ToolsPlan(
        objective="Run",
        steps=[],
        decision="finalize",
        decision_reason="done",
        stop_conditions=["done"],
        finalize_action="done",
    )

    calls = {"n": 0}

    def _plan_with_source(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return continue_plan, [], "planner"
        return finalize_plan, [], "planner"

    monkeypatch.setattr(orchestrator, "_plan_with_source", _plan_with_source)
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(
                planner_step_id="s1",
                batch_reason="parallel",
                requests=[{"question": "q0"}, {"question": "q1"}],
                continue_after=True,
                expected_progress="inspect",
            ),
            [],
        ),
    )

    class _OutOfOrderExecutor:
        def run_batch(self, *, run_id: str, requests, on_event=None):  # noqa: ANN001
            _ = (run_id, requests, on_event)
            return [
                BatchExecutionResult(
                    request_index=1,
                    ok=True,
                    response={
                        "answer": "second",
                        "sources": [],
                        "mode": "agent-tools",
                        "trace": [{"tool_name": "read_file", "status": "ok", "idx": 1}],
                        "warnings": [],
                    },
                    backend="redis",
                ),
                BatchExecutionResult(
                    request_index=0,
                    ok=True,
                    response={
                        "answer": "first",
                        "sources": [],
                        "mode": "agent-tools",
                        "trace": [{"tool_name": "semantic_search", "status": "ok", "idx": 0}],
                        "warnings": [],
                    },
                    backend="redis",
                ),
            ]

    orchestrator.executor = _OutOfOrderExecutor()

    result = orchestrator.run(
        request="execute",
        flow_context=None,
        index_dir=tmp_path / ".mana_index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=2,
    )
    assert result.toolsmanager_requests_count == 2
    assert result.execution_requests_ok == 2
    assert result.execution_requests_failed == 0
    assert len(result.trace) == 2
    assert result.trace[0].get("idx") == 0
    assert result.trace[1].get("idx") == 1


def test_tools_manager_mixed_batch_failures_are_warnings_not_crash(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)

    continue_plan = ToolsPlan(
        objective="Run",
        steps=[
            {
                "id": "s1",
                "title": "Inspect",
                "tool_intent": "inspect",
                "args_hint": "",
                "success_signal": "",
                "fallback": "",
                "status": "in_progress",
            }
        ],
        current_step_id="s1",
        decision="continue",
        stop_conditions=["done"],
        finalize_action="done",
    )
    finalize_plan = ToolsPlan(
        objective="Run",
        steps=[],
        decision="finalize",
        decision_reason="done",
        stop_conditions=["done"],
        finalize_action="done",
    )

    calls = {"n": 0}

    def _plan_with_source(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return continue_plan, [], "planner"
        return finalize_plan, [], "planner"

    monkeypatch.setattr(orchestrator, "_plan_with_source", _plan_with_source)
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(
                planner_step_id="s1",
                batch_reason="parallel",
                requests=[{"question": "q0"}, {"question": "q1"}],
                continue_after=True,
                expected_progress="inspect",
            ),
            [],
        ),
    )

    class _MixedExecutor:
        def run_batch(self, *, run_id: str, requests, on_event=None):  # noqa: ANN001
            _ = (run_id, requests, on_event)
            return [
                BatchExecutionResult(
                    request_index=0,
                    ok=False,
                    error_code="job_timeout",
                    error_message="timed out",
                    backend="redis",
                ),
                BatchExecutionResult(
                    request_index=1,
                    ok=True,
                    response={
                        "answer": "ok",
                        "sources": [],
                        "mode": "agent-tools",
                        "trace": [{"tool_name": "read_file", "status": "ok"}],
                        "warnings": [],
                    },
                    backend="redis",
                ),
            ]

    orchestrator.executor = _MixedExecutor()

    result = orchestrator.run(
        request="execute",
        flow_context=None,
        index_dir=tmp_path / ".mana_index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=2,
    )
    assert result.toolsmanager_requests_count == 2
    assert result.execution_requests_ok == 1
    assert result.execution_requests_failed == 1
    assert any("job_timeout" in str(item) for item in result.warnings)


def test_deterministic_fallback_edit_directive_mentions_write_retry_and_change_evidence(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    plan = ToolsPlan(
        objective="Update README section",
        steps=[
            {
                "id": "s1",
                "title": "Edit README",
                "tool_intent": "edit",
                "args_hint": "insert diagram section",
                "success_signal": "README updated",
                "fallback": "write_file fallback",
                "status": "in_progress",
            }
        ],
        current_step_id="s1",
        decision="continue",
        stop_conditions=["done"],
        finalize_action="done",
    )

    req = orchestrator._deterministic_fallback_request(
        request="insert project diagram into README.md",
        flow_context=None,
        plan=plan,
        step=plan.steps[0],
        pass_index=1,
    )

    assert req is not None
    text = str(req.question).lower()
    assert "apply_patch" in text
    assert "write_file" in text
    assert "changed_files" in text
    assert "conversational terminal" in text
