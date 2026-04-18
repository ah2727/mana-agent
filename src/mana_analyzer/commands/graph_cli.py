from __future__ import annotations

from .cli_internal import *
from .output import build_output_sink

@app.command()
def graph(
    path: str,
    output_dot: str | None = typer.Option(None, "--dot"),
    output_graphml: str | None = typer.Option(None, "--graphml"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    output_file = _resolve_output_file(path)
    sink = build_output_sink(command_name="graph", json_mode=as_json, output_file=output_file, console=console)
    logger.info("Graph command started", extra={"path": path, "output_dot": output_dot, "output_graphml": output_graphml})
    try:
        report = build_dependency_service().analyze(path)
        logger.debug("Graph dependency analyze completed", extra={"path": path})
    except Exception as exc:
        _log_exception("graph_deps_analyze", exc, path=path)
        raise

    try:
        if output_dot:
            Path(output_dot).write_text(report.to_dot(), encoding="utf-8")
            logger.info("Wrote graph DOT", extra={"output_dot": output_dot})
        if output_graphml:
            Path(output_graphml).write_text(report.to_graphml(), encoding="utf-8")
            logger.info("Wrote graph GraphML", extra={"output_graphml": output_graphml})
    except Exception as exc:
        _log_exception("graph_write_outputs", exc, output_dot=output_dot, output_graphml=output_graphml)
        raise

    payload = {
        "project_root": report.project_root,
        "module_edges": len(report.module_edges),
        "external_edges": len(report.dependency_edges),
        "dot_output": output_dot or "",
        "graphml_output": output_graphml or "",
    }
    if as_json:
        sink.emit_json(payload)
    else:
        sink.emit_kv(
            title="Graph Summary",
            items=[(key, str(value)) for key, value in payload.items()],
        )
