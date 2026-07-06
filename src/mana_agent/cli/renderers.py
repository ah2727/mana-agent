from __future__ import annotations

import json
from typing import Any

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from mana_agent.cli.events import ChatEvent
from mana_agent.telemetry.tokens import TokenUsage, TokenUsageTracker


def format_token_usage(usage: TokenUsage) -> str:
    prefix = "~" if usage.estimated and usage.total_tokens else ""
    if usage.total_tokens <= 0:
        return "tokens: unavailable"
    parts = [
        f"in {prefix}{usage.input_tokens}",
        f"out {prefix}{usage.output_tokens}",
    ]
    if usage.cached_input_tokens:
        parts.append(f"cached {prefix}{usage.cached_input_tokens}")
    if usage.reasoning_tokens:
        parts.append(f"reasoning {prefix}{usage.reasoning_tokens}")
    if usage.tool_result_tokens:
        parts.append(f"tools {prefix}{usage.tool_result_tokens}")
    parts.append(f"total {prefix}{usage.total_tokens}")
    return "tokens: " + " · ".join(parts)


def _status_icon(status: str, *, plain: bool = False) -> str:
    normalized = str(status or "").lower()
    if plain:
        return {
            "queued": "-",
            "running": ">",
            "success": "ok",
            "done": "ok",
            "failed": "x",
            "failure": "x",
            "skipped": "skip",
        }.get(normalized, "-")
    return {
        "queued": "•",
        "running": "⠙",
        "success": "✓",
        "done": "✓",
        "failed": "✗",
        "failure": "✗",
        "skipped": "↷",
    }.get(normalized, "•")


def _event_actor_label(event: ChatEvent) -> str:
    if event.subagent_id:
        return str(event.subagent_id)
    if str(event.agent_id or "").startswith("subagent_"):
        return str(event.agent_id)
    role = str(event.metadata.get("agent_role") or event.metadata.get("role") or "").strip()
    return role or "-"


class EventRenderer:
    def __init__(self, *, mode: str = "rich", trace_mode: str = "compact") -> None:
        self.mode = self.normalize_mode(mode)
        self.trace_mode = self.normalize_trace_mode(trace_mode)

    @staticmethod
    def normalize_mode(mode: str) -> str:
        value = str(mode or "rich").strip().lower()
        return value if value in {"fullscreen", "rich", "compact", "plain", "json"} else "rich"

    @staticmethod
    def normalize_trace_mode(mode: str) -> str:
        value = str(mode or "compact").strip().lower()
        return value if value in {"off", "compact", "full", "logs"} else "compact"

    def format_usage(self, usage: TokenUsage | None) -> str:
        return format_token_usage(usage or TokenUsage()).removeprefix("tokens: ")

    def render_event(self, event: ChatEvent) -> Any:
        if self.mode == "json":
            return json.dumps(event.as_dict(), ensure_ascii=False)
        if self.mode == "plain":
            return self._plain_event(event)
        if self.mode == "compact":
            return self._compact_event(event)
        return self._rich_event(event)

    def _plain_event(self, event: ChatEvent) -> str:
        duration = f"{event.duration_ms / 1000:.1f}s" if event.duration_ms else ""
        message = f" - {event.message}" if event.message else ""
        return f"[{_status_icon(event.status, plain=True)}] {event.step_id or event.event_id} {event.title}{message} {duration}".strip()

    def _compact_event(self, event: ChatEvent) -> Text:
        duration = f"{event.duration_ms / 1000:.1f}s" if event.duration_ms else ""
        text = Text()
        text.append(_status_icon(event.status), style="green" if event.status in {"success", "done"} else "cyan")
        text.append(f" {event.step_id or event.event_id[-6:]} ", style="dim")
        text.append(event.title or event.type, style="bold")
        if event.message:
            text.append(f" - {event.message}", style="dim")
        if duration:
            text.append(f" {duration}", style="dim")
        return text

    def _rich_event(self, event: ChatEvent) -> Panel:
        body = Table.grid(padding=(0, 1), expand=True)
        body.add_column(justify="right", no_wrap=True, style="bold")
        body.add_column(ratio=1, overflow="fold")
        body.add_row("status", f"{_status_icon(event.status)} {event.status}")
        if event.duration_ms:
            body.add_row("time", f"{event.duration_ms / 1000:.1f}s")
        token_text = self.format_usage(event.token_usage)
        if token_text != "unavailable":
            body.add_row("tokens", token_text)
        if event.message:
            label = "decision" if event.type == "agent.decision" else "summary"
            body.add_row(label, event.message)
        if self.trace_mode == "full" and event.metadata:
            body.add_row("metadata", json.dumps(event.metadata, ensure_ascii=False, default=str)[:1200])
        title = f"Step {event.step_id} · {event.title}" if event.step_id else event.title or event.type
        return Panel(body, title=title, title_align="left", border_style="cyan", box=box.ROUNDED)

    def render_events(self, events: list[ChatEvent], *, title: str = "Step timeline") -> Any:
        if self.mode == "json":
            return "\n".join(json.dumps(event.as_dict(), ensure_ascii=False) for event in events)
        if self.mode in {"plain", "compact"}:
            lines = [self.render_event(event) for event in events]
            return Group(*lines) if self.mode == "compact" else "\n".join(str(line) for line in lines)
        return Panel(Group(*(self._compact_event(event) for event in events)), title=title, box=box.ROUNDED)

    def render_tokens(self, tracker: TokenUsageTracker) -> Any:
        snapshot = tracker.snapshot()
        if self.mode == "json":
            return json.dumps({"type": "tokens", **snapshot}, ensure_ascii=False)
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column(overflow="fold")
        current = tracker.by_turn.get(tracker.current_turn_id, TokenUsage())
        table.add_row("turn", self.format_usage(current))
        table.add_row("session", self.format_usage(tracker.session_total))
        table.add_row("cache", f"read {tracker.session_total.cached_input_tokens} · write {tracker.session_total.cache_creation_tokens}")
        table.add_row("subagents", str(sum(item.total_tokens for item in tracker.by_subagent.values())))
        table.add_row("tools injected", str(sum(item.tool_result_tokens for item in tracker.by_tool_result.values())))
        table.add_row("accounting", "estimated values are prefixed with ~; exact values require provider usage")
        if self.mode == "fullscreen":
            max_total = max(1, tracker.session_total.total_tokens)
            table.add_row("turn bar", ProgressBar(total=max_total, completed=min(max_total, current.total_tokens), width=24))
            table.add_row("session bar", ProgressBar(total=max_total, completed=tracker.session_total.total_tokens, width=24))
            for step_id, usage in list(tracker.by_step.items())[-5:]:
                table.add_row(
                    f"step {step_id}",
                    ProgressBar(total=max_total, completed=min(max_total, usage.total_tokens), width=24),
                )
        return Panel(table, title="Token usage", border_style="magenta", box=box.ROUNDED)

    def render_tool_activity(self, events: list[ChatEvent]) -> Any:
        tool_events = [event for event in events if event.type.startswith("tool.")]
        if self.mode == "json":
            return "\n".join(json.dumps(event.as_dict(), ensure_ascii=False) for event in tool_events)
        table = Table(show_header=True, header_style="bold", box=box.SIMPLE, expand=True)
        table.add_column("", no_wrap=True)
        table.add_column("Tool", no_wrap=True)
        table.add_column("Subagent", no_wrap=True)
        table.add_column("Purpose", overflow="fold")
        table.add_column("Duration", justify="right", no_wrap=True)
        table.add_column("Result", overflow="fold")
        for event in tool_events[-30:]:
            table.add_row(
                _status_icon(event.status, plain=self.mode == "plain"),
                str(event.metadata.get("tool_name") or event.title or "tool"),
                _event_actor_label(event),
                str(event.metadata.get("args_summary") or event.message or "-"),
                f"{event.duration_ms / 1000:.1f}s" if event.duration_ms else "",
                str(event.metadata.get("result_summary") or event.message or "-"),
            )
        return Panel(table, title="Tool activity", border_style="cyan", box=box.ROUNDED)

    def render_subagents(self, events: list[ChatEvent]) -> Any:
        subagent_events = [event for event in events if event.type.startswith("subagent.")]
        if self.mode == "json":
            return "\n".join(json.dumps(event.as_dict(), ensure_ascii=False) for event in subagent_events)
        table = Table(show_header=True, header_style="bold", box=box.SIMPLE, expand=True)
        table.add_column("ID", no_wrap=True)
        table.add_column("Role", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        table.add_column("Current step", overflow="fold")
        table.add_column("Tokens", justify="right", no_wrap=True)
        table.add_column("Summary", overflow="fold")
        latest: dict[str, ChatEvent] = {}
        for event in subagent_events:
            key = str(event.subagent_id or event.agent_id or event.event_id)
            latest[key] = event
        for key, event in latest.items():
            table.add_row(
                key,
                str(event.metadata.get("role") or event.title or "subagent"),
                event.status,
                str(event.metadata.get("current_step") or event.step_id or "-"),
                str(event.token_usage.total_tokens if event.token_usage else 0),
                event.message or "-",
            )
        return Panel(table, title="Subagents", border_style="green", box=box.ROUNDED)

    def render_log_lines(self, lines: list[str]) -> Any:
        if self.mode == "json":
            return json.dumps({"type": "trace.logs", "lines": lines}, ensure_ascii=False)
        body = "\n".join(lines[-40:]) if lines else "No trace log lines available."
        if self.mode == "plain":
            return "Trace logs\n" + body
        return Panel(body, title="Trace logs", border_style="yellow", box=box.ROUNDED)
