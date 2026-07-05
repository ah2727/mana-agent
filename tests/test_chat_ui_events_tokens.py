from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from mana_agent.cli.chat_ui import ChatUIState, render_startup_header
from mana_agent.cli.events import make_event
from mana_agent.cli.renderers import EventRenderer
from mana_agent.telemetry.tokens import TokenUsageTracker, token_usage_from_provider


def _render_to_text(renderable) -> str:
    console = Console(record=True, width=100)
    console.print(renderable)
    return console.export_text()


def test_token_usage_tracker_records_exact_provider_usage() -> None:
    tracker = TokenUsageTracker()
    tracker.start_turn("turn-1")
    usage = tracker.record_model_call(
        "call-1",
        usage={
            "input_tokens": 10,
            "output_tokens": 4,
            "total_tokens": 20,
            "input_token_details": {"cached_tokens": 3, "cache_creation_tokens": 2},
            "output_token_details": {"reasoning_tokens": 6},
        },
        provider="openai",
        model="gpt-test",
        agent_id="main",
        step_id="05",
    )

    assert usage.estimated is False
    assert usage.input_tokens == 10
    assert usage.output_tokens == 4
    assert usage.reasoning_tokens == 6
    assert tracker.by_turn["turn-1"].total_tokens == 20
    assert tracker.by_provider_model["openai:gpt-test"].cached_input_tokens == 3


def test_token_usage_tracker_marks_missing_provider_usage_estimated() -> None:
    tracker = TokenUsageTracker()
    usage = tracker.record_model_call("call-1", usage=None, estimated_text="hello world")

    assert usage.estimated is True
    assert usage.total_tokens > 0


def test_provider_usage_object_normalizes_openai_style_fields() -> None:
    class Usage:
        input_tokens = 7
        output_tokens = 5
        total_tokens = 12
        input_token_details = {"cached_tokens": 2}
        output_token_details = {"reasoning_tokens": 1}

    usage = token_usage_from_provider(Usage())

    assert usage.input_tokens == 7
    assert usage.output_tokens == 5
    assert usage.cached_input_tokens == 2
    assert usage.reasoning_tokens == 1
    assert usage.estimated is False


def test_event_schema_contains_required_fields() -> None:
    event = make_event(
        "tool.finished",
        title="read_file",
        message="Read src/app.py",
        status="success",
        session_id="sess-1",
        turn_id="turn-1",
        agent_id="main",
        step_id="08",
        metadata={"path": "src/app.py"},
    ).finish(status="success")
    data = event.as_dict()

    for key in (
        "event_id",
        "parent_event_id",
        "session_id",
        "turn_id",
        "agent_id",
        "subagent_id",
        "step_id",
        "type",
        "status",
        "title",
        "message",
        "started_at",
        "ended_at",
        "duration_ms",
        "token_usage",
        "metadata",
    ):
        assert key in data
    assert data["type"] == "tool.finished"
    assert data["token_usage"]["estimated"] is False


def test_event_renderer_modes_render_without_raw_json_noise() -> None:
    event = make_event(
        "agent.decision",
        title="Agent decision",
        message="Decision summary: inspect CLI renderer before editing.",
        status="success",
        step_id="05",
    ).finish(status="success")

    rich_text = _render_to_text(EventRenderer(mode="rich").render_event(event))
    compact_text = _render_to_text(EventRenderer(mode="compact").render_event(event))
    plain_text = str(EventRenderer(mode="plain").render_event(event))
    json_text = str(EventRenderer(mode="json").render_event(event))

    assert "Agent decision" in rich_text
    assert "Agent decision" in compact_text
    assert "inspect CLI renderer" in plain_text
    assert json.loads(json_text)["type"] == "agent.decision"


def test_tools_and_subagents_render_from_events_only() -> None:
    tool_event = make_event(
        "tool.failed",
        title="run_tests",
        message="pytest failed",
        status="failed",
        step_id="09",
        metadata={"tool_name": "run_tests", "args_summary": "pytest tests/test_cli_ui.py", "result_summary": "1 failed"},
    ).finish(status="failed")
    subagent_event = make_event(
        "subagent.finished",
        title="test-runner",
        message="Tests passed",
        status="success",
        agent_id="main",
        subagent_id="A-003",
        metadata={"role": "test-runner", "current_step": "verification"},
    ).finish(status="success")
    renderer = EventRenderer(mode="rich")

    tool_text = _render_to_text(renderer.render_tool_activity([tool_event]))
    subagent_text = _render_to_text(renderer.render_subagents([subagent_event]))

    assert "run_tests" in tool_text
    assert "pytest tests/test_cli_ui.py" in tool_text
    assert "A-003" in subagent_text
    assert "test-runner" in subagent_text


def test_chat_ui_startup_header_and_token_command_render() -> None:
    console = Console(record=True, width=100)
    state = ChatUIState(
        repo_root=Path.cwd(),
        provider="openai",
        model="gpt-test",
        tools_enabled=True,
        memory_enabled=True,
        skills_status="indexed",
        ui_mode="rich",
    )
    render_startup_header(console, state)
    state.tracker.start_turn("turn-1")
    state.tracker.record_tool_result("tool-1", "some result text", turn_id="turn-1")
    console.print(state.renderer.render_tokens(state.tracker))
    rendered = console.export_text()

    assert "Mana-Agent" in rendered
    assert "/tokens" in rendered
    assert "Token usage" in rendered
    assert "~" in rendered
