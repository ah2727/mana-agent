from __future__ import annotations

from .cli_internal import *
from .output import build_output_sink

@app.command("flow")
def flow_cmd(
    project_path: str | None = typer.Argument(
        None,
        help="Project root containing .mana/index/chat_memory.sqlite3.",
    ),
    flow_id: str | None = typer.Option(None, "--flow-id", help="Flow ID to inspect; defaults to active flow."),
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
    max_turns: int = typer.Option(5, "--max-turns", help="Maximum recent turns to include."),
    max_tasks: int = typer.Option(20, "--max-tasks", help="Maximum open tasks to include."),
) -> None:
    resolved_project_path = project_path or "."
    output_file = _resolve_output_file(resolved_project_path)
    sink = build_output_sink(
        command_name="flow",
        json_mode=(str(output_format or "text").strip().lower() == "json"),
        output_file=output_file,
        console=console,
    )
    project_root = Path(resolved_project_path).resolve()
    resolved_format = str(output_format or "text").strip().lower()
    if resolved_format not in {"text", "json"}:
        raise typer.BadParameter("--format must be 'text' or 'json'.")

    memory_service = CodingMemoryService(
        project_root=project_root,
        max_turns=max(1, int(max_turns)),
        max_tasks=max(1, int(max_tasks)),
    )

    target_flow_id = flow_id or memory_service.get_active_flow_id()
    if not target_flow_id:
        sink.emit_warning("No active coding flow found.")
        return

    summary_payload = _build_flow_summary_payload(memory_service, str(target_flow_id))
    if summary_payload is None:
        sink.emit_warning(f"Flow not found: {target_flow_id}")
        return

    if resolved_format == "json":
        sink.emit_json(summary_payload)
        return

    capture_console = Console(record=True)
    _render_flow_summary(
        capture_console,
        summary_payload,
        include_checklist=True,
        include_transitions=True,
        include_recent_turns=True,
    )
    sink.emit_text(capture_console.export_text().rstrip())
