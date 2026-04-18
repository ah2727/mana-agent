from __future__ import annotations

from .cli_internal import *
from .output import build_output_sink
from mana_analyzer.renderers.html_report import render_describe_html


def _describe_payload(report: object) -> dict:
    if isinstance(report, dict):
        return report
    if hasattr(report, "to_dict"):
        return report.to_dict()
    return {}


def _render_describe_markdown_from_payload(payload: dict) -> str:
    descriptions = payload.get("descriptions") or payload.get("files") or []
    lines = ["# Repository Description", "", "## Architecture", ""]
    lines.append(str(payload.get("architecture_summary") or "Architecture summary unavailable."))
    lines.extend(["", "## Technology", ""])
    lines.append(str(payload.get("tech_summary") or "Technology summary unavailable."))
    lines.extend(["", "## File Summaries", ""])
    if descriptions:
        for item in descriptions:
            file_path = item.get("file_path") or item.get("path") or "unknown"
            language = item.get("language") or "unknown"
            summary = item.get("summary") or "No summary available."
            lines.append(f"- `{file_path}` ({language}) - {summary}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


@app.command()
def describe(
    path: str,
    use_llm: bool = typer.Option(True, "--llm/--no-llm"),
    model: str | None = typer.Option(None, "--llm-model"),
    max_files: int = typer.Option(12, "--max-files"),
    functions: bool = typer.Option(False, "--functions"),
    include: str | None = typer.Option(None, "--include", help="Comma-separated include glob patterns."),
    exclude: str | None = typer.Option(None, "--exclude", help="Comma-separated exclude glob patterns."),
    recent_days: int | None = typer.Option(None, "--recent-days", help="Only analyze files modified in last N days."),
    include_docstrings: bool = typer.Option(True, "--docstrings/--no-docstrings"),
    no_cache: bool = typer.Option(False, "--no-cache"),
    output_format: str = typer.Option("all", "--output-format"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    output_file = _resolve_output_file(path)
    sink = build_output_sink(command_name="describe", json_mode=as_json, output_file=output_file, console=console)

    if output_format not in {"json", "markdown", "html", "all"}:
        raise typer.BadParameter("--output-format must be one of: json, markdown, html, all")

    logger.info(
        "Describe command started",
        extra={
            "path": path,
            "use_llm": use_llm,
            "model_override": model,
            "max_files": max_files,
            "functions": functions,
            "include": include,
            "exclude": exclude,
            "recent_days": recent_days,
            "include_docstrings": include_docstrings,
            "no_cache": no_cache,
            "output_format": output_format,
        },
    )

    settings = Settings() if use_llm else None
    if use_llm:
        assert settings is not None
        try:
            service = build_describe_service(settings, model_override=model, use_llm=True)
        except TypeError:
            service = build_describe_service(
                dependency_service=build_dependency_service(),
                llm_chain=None,
                include_tests=False,
            )
    else:
        service = build_describe_service(
            dependency_service=build_dependency_service(),
            llm_chain=None,
            include_tests=False,
        )

    include_patterns = [item.strip() for item in (include or "").split(",") if item.strip()] or None
    exclude_patterns = [item.strip() for item in (exclude or "").split(",") if item.strip()] or None
    modified_since = None
    if recent_days is not None:
        if recent_days < 0:
            raise typer.BadParameter("--recent-days must be >= 0")
        modified_since = datetime.now() - timedelta(days=recent_days)

    try:
        try:
            report = service.describe(
                path,
                max_files=max_files,
                include_functions=functions,
                use_llm=use_llm,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                modified_since=modified_since,
                include_docstrings=include_docstrings,
                use_cache=not no_cache,
            )
        except TypeError:
            report = service.describe(path)
        logger.info("Describe completed", extra={"path": path})
    except Exception as exc:
        _log_exception("describe_service.describe", exc, path=path)
        raise

    payload = _describe_payload(report)
    if hasattr(service, "render_markdown"):
        markdown = service.render_markdown(report)
    else:
        markdown = _render_describe_markdown_from_payload(payload)
    html = render_describe_html(payload, markdown)

    out_json, out_md, out_html = _resolve_describe_artifact_paths(path)
    if output_format in {"json", "all"}:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if output_format in {"markdown", "all"}:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(markdown, encoding="utf-8")
    if output_format in {"html", "all"}:
        out_html.parent.mkdir(parents=True, exist_ok=True)
        out_html.write_text(html, encoding="utf-8")

    if as_json:
        sink.emit_json(payload)
        return

    sink.emit_text(markdown)
    sink.emit_kv(
        title="Describe Artifacts",
        items=[
            ("json", str(out_json) if output_format in {"json", "all"} else "(skipped)"),
            ("markdown", str(out_md) if output_format in {"markdown", "all"} else "(skipped)"),
            ("html", str(out_html) if output_format in {"html", "all"} else "(skipped)"),
        ],
    )
