from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mana_agent.cli.chat_ui import ChatUIState

try:
    from prompt_toolkit.application import Application
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.shortcuts import radiolist_dialog
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import Box, Frame, TextArea

    _PTK_IMPORT_OK = True
except Exception:  # pragma: no cover - exercised only when prompt_toolkit missing
    _PTK_IMPORT_OK = False


@dataclass(frozen=True, slots=True)
class MenuOption:
    value: str
    label: str
    aliases: tuple[str, ...] = ()


def fullscreen_available() -> bool:
    if not _PTK_IMPORT_OK:
        return False
    if os.getenv("CI"):
        return False
    if str(os.getenv("TERM", "") or "").strip().lower() in {"", "dumb"}:
        return False
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except Exception:
        return False


def animation_enabled() -> bool:
    raw = str(os.getenv("MANA_CHAT_ANIMATION", "1") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"} and fullscreen_available()


def show_startup_pet_animation() -> None:
    if not animation_enabled():
        return
    frames = (
        r" /\_/\   Mana is waking up",
        r"( o.o )  Loading repo context",
        r" > ^ <   Ready for agent work",
    )
    for frame in frames:
        sys.stdout.write("\r" + frame + " " * 12)
        sys.stdout.flush()
        time.sleep(0.12)
    sys.stdout.write("\r" + " " * 48 + "\r")
    sys.stdout.flush()


def _clip(text: Any, limit: int) -> str:
    raw = str(text or "").replace("\n", " ").strip()
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 1)].rstrip() + "…"


def token_bar(value: int, maximum: int | None = None, *, width: int = 18) -> str:
    value = max(0, int(value or 0))
    if maximum is None or int(maximum or 0) <= 0:
        maximum = max(value, 1)
    maximum = max(1, int(maximum))
    filled = min(width, int(round((value / maximum) * width))) if value else 0
    return "[" + ("#" * filled).ljust(width, "-") + f"] {value}/{maximum}"


def _event_actor_label(event: Any) -> str:
    subagent_id = str(getattr(event, "subagent_id", "") or "").strip()
    if subagent_id:
        return subagent_id
    agent_id = str(getattr(event, "agent_id", "") or "").strip()
    if agent_id.startswith("subagent_"):
        return agent_id
    metadata = getattr(event, "metadata", {}) or {}
    role = str(metadata.get("agent_role") or metadata.get("role") or "").strip()
    return role or agent_id or "main"


def _event_model_label(event: Any) -> str:
    metadata = getattr(event, "metadata", {}) or {}
    level = str(metadata.get("model_level") or "").strip()
    model = str(metadata.get("resolved_model") or "").strip()
    if level and model:
        return f"{level} • {model}"
    return level or model


def _duration_label(event: Any) -> str:
    duration_ms = getattr(event, "duration_ms", None)
    if duration_ms is None:
        return ""
    milliseconds = int(duration_ms)
    return f"{milliseconds}ms" if milliseconds < 1000 else f"{milliseconds / 1000:.1f}s"


def _conversation_text(state: "ChatUIState") -> str:
    rows: list[str] = []
    for item in state.conversation[-12:]:
        role = str(item.get("role", "") or "").strip().lower()
        content = _clip(str(item.get("content", "") or "").strip(), 160)
        if not content:
            continue
        if role == "user":
            rows.append(f"you: {content}")
        elif role == "assistant":
            rows.append(f"assistant: {content}")
        else:
            rows.append(content)
    if rows:
        return "\n".join(rows[-12:])
    for event in state.events[-18:]:
        if event.type == "turn.started":
            rows.append(f"> {event.message or event.title}")
        elif event.type == "agent.decision":
            rows.append(f"agent: {_clip(event.message or event.title, 120)}")
        elif event.type == "turn.finished":
            rows.append(f"done: {_clip(event.message or event.title, 120)}")
    return "\n".join(rows[-12:]) or "No messages yet."


def _steps_text(state: "ChatUIState") -> str:
    rows: list[str] = []
    for event in state.events[-28:]:
        if event.type.startswith("tool.") or event.type.startswith("subagent."):
            continue
        label = event.step_id or event.type
        rows.append(f"{label:>4} {event.status:<8} {_clip(event.title, 28)} {_clip(event.message, 80)}")
    return "\n".join(rows[-14:]) or "Waiting for the first request."


def _tools_text(state: "ChatUIState") -> str:
    rows: list[str] = []
    grouped: dict[str, list[Any]] = {}
    for event in state.tool_runs[-20:]:
        grouped.setdefault(_event_actor_label(event), []).append(event)
    for actor, events in grouped.items():
        model = next((label for label in (_event_model_label(item) for item in events) if label), "")
        header = actor if not model else f"{actor} • {model}"
        rows.append(_clip(header, 120))
        for index, event in enumerate(events):
            metadata = event.metadata or {}
            branch = "└─" if index == len(events) - 1 else "├─"
            tool = metadata.get("tool_name") or event.title or "tool"
            detail = metadata.get("args_summary") or metadata.get("result_summary") or event.message or ""
            status = "✓" if event.status in {"success", "done"} else ("✗" if event.status in {"failed", "failure"} else "•")
            duration = _duration_label(event)
            suffix = f" {status}{(' ' + duration) if duration else ''}"
            if detail:
                suffix += f": {_clip(detail, 80)}"
            rows.append(f"  {branch} {_clip(tool, 24)}{suffix}")
    return "\n".join(rows[-14:]) or "No tool calls yet."


def _agents_text(state: "ChatUIState") -> str:
    latest: dict[str, Any] = {}
    models: dict[str, str] = {}
    for event in state.subagent_events:
        key = str(event.subagent_id or event.agent_id or event.event_id)
        latest[key] = event
        if not models.get(key):
            models[key] = _event_model_label(event)
    rows = []
    for key, event in list(latest.items())[-10:]:
        role = event.metadata.get("role") or event.metadata.get("agent_role") or event.title or "subagent"
        model = models.get(key, "") or _event_model_label(event)
        rows.append(_clip(key, 80))
        if model:
            rows.append(f"  {model}")
        rows.append(f"  {event.status:<8} {_clip(role, 22)} {_clip(event.message, 80)}")
    return "\n".join(rows[-12:]) or "No subagents active."


def _tokens_text(state: "ChatUIState") -> str:
    current = state.tracker.by_turn.get(state.tracker.current_turn_id)
    session = state.tracker.session_total
    current_total = int(getattr(current, "total_tokens", 0) or 0)
    session_total = int(getattr(session, "total_tokens", 0) or 0)
    step_rows = []
    for step_id, usage in list(state.tracker.by_step.items())[-6:]:
        step_rows.append(f"{step_id:<8} {token_bar(usage.total_tokens, max(session_total, usage.total_tokens, 1), width=12)}")
    rows = [
        f"turn    {token_bar(current_total, max(session_total, current_total, 1))}",
        f"session {token_bar(session_total, max(session_total, current_total, 1))}",
        f"tools   {session.tool_result_tokens}",
        f"subagents {sum(item.total_tokens for item in state.tracker.by_subagent.values())}",
    ]
    if step_rows:
        rows.append("")
        rows.extend(step_rows)
    return "\n".join(rows)


def _style() -> "Style":
    return Style.from_dict(
        {
            "frame.label": "bold #5fd7ff",
            "header": "bold #5fd7ff",
            "footer": "#bcbcbc bg:#1c1c1c",
            "input": "#ffffff bg:#262626",
        }
    )


def read_fullscreen_chat_input(state: "ChatUIState", *, prompt: str = "mana") -> str:
    if not fullscreen_available():
        raise RuntimeError("full-screen chat is unavailable")

    result: dict[str, str] = {"text": ""}
    kb = KeyBindings()

    @kb.add("c-c")
    def _cancel(event) -> None:
        event.app.exit(exception=KeyboardInterrupt)

    @kb.add("c-d")
    def _eof(event) -> None:
        event.app.exit(exception=EOFError)

    @kb.add("enter", eager=True)
    def _submit(event) -> None:
        event.current_buffer.validate_and_handle()

    @kb.add("c-m", eager=True)
    def _submit_ctrl_m(event) -> None:
        event.current_buffer.validate_and_handle()

    @kb.add("c-j", eager=True)
    def _newline(event) -> None:
        event.current_buffer.insert_text("\n")

    @kb.add("escape", "enter", eager=True)
    def _alt_enter_newline(event) -> None:
        event.current_buffer.insert_text("\n")

    def _accept(buffer) -> bool:  # noqa: ANN001
        result["text"] = str(buffer.text or "").strip()
        app.exit(result=result["text"])
        return True

    input_box = TextArea(
        height=5,
        multiline=True,
        accept_handler=_accept,
        prompt=f"{prompt} > ",
        style="class:input",
    )
    header = Window(
        FormattedTextControl(
            HTML(
                f"<header>Mana-Agent</header>  repo={state.repo_root.name}  "
                f"model={state.model}  ui=fullscreen"
            )
        ),
        height=1,
    )
    footer = Window(
        FormattedTextControl(
            "Enter send | Esc+Enter or Ctrl+J newline | Ctrl+C cancel | Ctrl+D exit | /ui rich to leave full-screen"
        ),
        height=1,
        style="class:footer",
    )
    main = HSplit(
        [
            header,
            VSplit(
                [
                    Frame(Window(FormattedTextControl(lambda: _conversation_text(state)), wrap_lines=True), title="Chat"),
                    Frame(Window(FormattedTextControl(lambda: _steps_text(state)), wrap_lines=True), title="Steps"),
                ]
            ),
            VSplit(
                [
                    Frame(Window(FormattedTextControl(lambda: _tools_text(state)), wrap_lines=True), title="Tools"),
                    Frame(Window(FormattedTextControl(lambda: _agents_text(state)), wrap_lines=True), title="Subagents"),
                    Frame(Window(FormattedTextControl(lambda: _tokens_text(state)), wrap_lines=True), title="Tokens"),
                ],
                height=10,
            ),
            Frame(Box(input_box, padding=1), title="Message"),
            footer,
        ]
    )
    app = Application(layout=Layout(main, focused_element=input_box), key_bindings=kb, full_screen=True, style=_style())
    text = app.run()
    return str(text or result["text"]).strip()


def run_fullscreen_worker(
    state: "ChatUIState",
    *,
    title: str,
    worker: Callable[[], Any],
) -> Any:
    if not fullscreen_available():
        return worker()

    result: dict[str, Any] = {}
    done = threading.Event()
    kb = KeyBindings()

    @kb.add("c-c")
    def _cancel(event) -> None:
        event.app.exit(exception=KeyboardInterrupt)

    def _complete() -> None:
        try:
            result["value"] = worker()
        except BaseException as exc:  # noqa: BLE001 - re-raised after app exits
            result["error"] = exc
        finally:
            done.set()
            while getattr(app, "loop", None) is None:
                time.sleep(0.01)
            app.loop.call_soon_threadsafe(app.exit)

    header = Window(
        FormattedTextControl(
            HTML(
                f"<header>Mana-Agent</header>  repo={state.repo_root.name}  "
                f"model={state.model}  ui=fullscreen"
            )
        ),
        height=1,
    )
    footer = Window(
        FormattedTextControl(lambda: f"{title} | tools update in the Tools pane | Ctrl+C cancel"),
        height=1,
        style="class:footer",
    )
    body = HSplit(
        [
            header,
            VSplit(
                [
                    Frame(Window(FormattedTextControl(lambda: _conversation_text(state)), wrap_lines=True), title="Chat"),
                    Frame(Window(FormattedTextControl(lambda: _steps_text(state)), wrap_lines=True), title="Steps"),
                ]
            ),
            VSplit(
                [
                    Frame(Window(FormattedTextControl(lambda: _tools_text(state)), wrap_lines=True), title="Tools"),
                    Frame(Window(FormattedTextControl(lambda: _agents_text(state)), wrap_lines=True), title="Subagents"),
                    Frame(Window(FormattedTextControl(lambda: _tokens_text(state)), wrap_lines=True), title="Tokens"),
                ],
                height=12,
            ),
            Frame(
                Window(
                    FormattedTextControl(
                        lambda: "Working..." if not done.is_set() else "Completed. Returning to chat input."
                    ),
                    wrap_lines=True,
                ),
                title="Status",
                height=5,
            ),
            footer,
        ]
    )
    app = Application(layout=Layout(body), key_bindings=kb, full_screen=True, style=_style(), refresh_interval=0.2)
    thread = threading.Thread(target=_complete, daemon=True)
    thread.start()
    app.run()
    thread.join(timeout=0.1)
    if "error" in result:
        raise result["error"]
    return result.get("value")


def select_option(
    *,
    title: str,
    text: str,
    options: Iterable[MenuOption],
    input_func: Callable[[str], str] | None = None,
) -> str:
    items = list(options)
    if not items:
        return ""
    if fullscreen_available() and input_func is None:
        result = radiolist_dialog(
            title=title,
            text=text,
            values=[(item.value, item.label) for item in items],
            style=_style(),
        ).run()
        return str(result or "")

    prompt_lines = [text, ""]
    for index, item in enumerate(items, start=1):
        prompt_lines.append(f"{index}. {item.label}")
    prompt_lines.append("")
    raw = (input_func or input)("\n".join(prompt_lines) + "Enter choice: ")
    normalized = str(raw or "").strip().lower()
    if not normalized:
        return ""
    if normalized.isdigit():
        index = int(normalized) - 1
        if 0 <= index < len(items):
            return items[index].value
    for item in items:
        aliases = {item.value.lower(), item.label.lower(), *(alias.lower() for alias in item.aliases)}
        if normalized in aliases:
            return item.value
    return normalized
