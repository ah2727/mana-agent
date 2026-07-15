"""TUI auto-chat tool event emission / replay."""

from __future__ import annotations

from pathlib import Path

from mana_agent.chat.events import ToolCallEvent, ToolResultEvent
from mana_agent.chat.history import ChatHistory
from mana_agent.gateway.turn_engine import ChatTurnResult, _serialize_tool_traces
from mana_agent.tui.app import ManaChatApp


def test_serialize_tool_traces_from_response_objects() -> None:
    class Trace:
        def to_dict(self):
            return {
                "tool_name": "web_search",
                "args_summary": "query=x",
                "duration_ms": 1.0,
                "status": "ok",
                "output_preview": "hit",
            }

    class Resp:
        trace = [Trace()]

    rows = _serialize_tool_traces(Resp())
    assert rows[0]["tool_name"] == "web_search"


def test_tui_replays_auto_chat_traces_into_history(tmp_path: Path) -> None:
    history = ChatHistory()
    app = ManaChatApp(history=history, repo_root=tmp_path, model="gpt-test")
    result = ChatTurnResult(
        answer="done",
        auto_chat_mode="answer_only",
        used_coding_agent=False,
        payload={
            "route": "auto_chat",
            "trace": [
                {
                    "tool_name": "email_read",
                    "args_summary": "message_ref=x",
                    "status": "ok",
                    "output_preview": "hello",
                    "duration_ms": 5,
                }
            ],
        },
        trace=[
            {
                "tool_name": "email_read",
                "args_summary": "message_ref=x",
                "status": "ok",
                "output_preview": "hello",
                "duration_ms": 5,
            }
        ],
    )
    app._replay_tool_traces_from_result(result, turn_id="turn-auto")
    tools = [e for e in history.get_events() if isinstance(e, (ToolCallEvent, ToolResultEvent))]
    assert any(isinstance(e, ToolCallEvent) and e.tool_name == "email_read" for e in tools)
    assert any(isinstance(e, ToolResultEvent) and e.tool_name == "email_read" and e.success for e in tools)


def test_tool_emit_bridge_writes_history(tmp_path: Path) -> None:
    history = ChatHistory()
    app = ManaChatApp(history=history, repo_root=tmp_path, model="gpt-test")
    app._tool_cid_map = {}
    bridge = app._make_tool_emit_bridge(original_emit=lambda *a, **k: None, turn_id="t1")
    bridge("start", "web_search", args='{"query":"x"}', event_id="e1")
    bridge("end", "web_search", duration=0.01, event_id="e1")
    names = [e.tool_name for e in history.get_events() if isinstance(e, (ToolCallEvent, ToolResultEvent))]
    assert names.count("web_search") == 2
