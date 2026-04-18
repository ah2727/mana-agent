from __future__ import annotations

import os
from .cli_internal import *
from .output import build_output_sink
from mana_analyzer.renderers.html_report import render_report_html

@app.command()
def report(
    path: str,
    model: str | None = typer.Option(None, "--model"),
    llm_max_files: int = typer.Option(10, "--llm-max-files"),
    summary_max_files: int = typer.Option(12, "--summary-max-files"),
    full_structure: bool = typer.Option(False, "--full-structure"),
    include_tests: bool = typer.Option(False, "--include-tests"),
    online: bool = typer.Option(True, "--online/--offline"),
    osv_timeout_seconds: int = typer.Option(10, "--osv-timeout-seconds"),
    security_scope: str = typer.Option("all", "--security-scope"),
    output_format: str = typer.Option("all", "--output-format"),
    json_out: str | None = typer.Option(None, "--json-out"),
    markdown_out: str | None = typer.Option(None, "--markdown-out"),
    html_out: str | None = typer.Option(None, "--html-out"),
    report_profile: str = typer.Option("standard", "--report-profile"),
    detail_line_target: int = typer.Option(350, "--detail-line-target"),
    security_lens: str = typer.Option("defensive-red-team", "--security-lens"),
    as_json: bool = typer.Option(False, "--json", help="Print full JSON report to console."),
) -> None:
    output_file = _resolve_output_file(path)
    sink = build_output_sink(command_name="report", json_mode=as_json, output_file=output_file, console=console)

    # Validate basic options
    if output_format not in {"json", "markdown", "html", "all"}:
        raise typer.BadParameter("--output-format must be one of: json, markdown, html, all")
    if security_scope not in {"all", "runtime", "dev"}:
        raise typer.BadParameter("--security-scope must be one of: all, runtime, dev")
    if osv_timeout_seconds <= 0:
        raise typer.BadParameter("--osv-timeout-seconds must be > 0")
    if report_profile not in {"standard", "deep"}:
        raise typer.BadParameter("--report-profile must be standard|deep")
    if security_lens not in {"defensive-red-team", "architecture", "compliance"}:
        raise typer.BadParameter(
            "--security-lens must be defensive-red-team|architecture|compliance"
        )

    # LLM is always enabled — resolve model from CLI flag or environment
    final_model = model or os.environ.get("LLM_MODEL")
    if not final_model:
        raise typer.BadParameter(
            "No LLM model configured. "
            "Set LLM_MODEL in your environment or pass --model."
        )

    # Deep profile forces full structure and clamps detail target
    if report_profile == "deep":
        detail_line_target = _clamp_detail_line_target(detail_line_target)
        full_structure = True

    # Resolve output paths
    report_json_default, report_md_default, report_html_default = _resolve_report_artifact_paths(path)
    out_json = _resolve_out_path(json_out, report_json_default, suffix=".json")
    out_md = _resolve_out_path(markdown_out, report_md_default, suffix=".md")
    out_html = _resolve_out_path(html_out, report_html_default, suffix=".html")

    logger.info(
        "Report command started",
        extra={
            "path": path,
            "model": final_model,
            "llm_max_files": llm_max_files,
            "summary_max_files": summary_max_files,
            "full_structure": full_structure,
            "include_tests": include_tests,
            "online": online,
            "osv_timeout_seconds": osv_timeout_seconds,
            "security_scope": security_scope,
            "output_format": output_format,
            "report_profile": report_profile,
            "detail_line_target": detail_line_target,
            "security_lens": security_lens,
            "out_json": str(out_json),
            "out_md": str(out_md),
            "out_html": str(out_html),
        },
    )

    # Build and invoke the report service — LLM always on
    service = build_report_service(
        use_llm=True,
        model_override=final_model,
        include_tests=include_tests,
    )

    try:
        report_obj = service.generate(
            target_path=path,
            with_llm=True,
            model_override=final_model,
            llm_max_files=llm_max_files,
            summary_max_files=summary_max_files,
            full_structure=full_structure,
            online=online,
            osv_timeout_seconds=osv_timeout_seconds,
            security_scope=security_scope,
            report_profile=report_profile,
            detail_line_target=detail_line_target,
            security_lens=security_lens,
        )
        logger.info("Report service generate completed", extra={"path": path})
    except Exception as exc:
        _log_exception("report_service.generate", exc, path=path)
        raise

    # Serialize and write outputs
    report_obj.meta.output_format = output_format
    payload = report_obj.to_dict()
    markdown = service.render_markdown(report_obj)
    html = render_report_html(payload, markdown)

    try:
        if output_format in {"json", "all"}:
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            logger.info("Wrote report JSON", extra={"out_json": str(out_json)})
        if output_format in {"markdown", "all"}:
            out_md.parent.mkdir(parents=True, exist_ok=True)
            out_md.write_text(markdown, encoding="utf-8")
            logger.info("Wrote report Markdown", extra={"out_md": str(out_md)})
        if output_format in {"html", "all"}:
            out_html.parent.mkdir(parents=True, exist_ok=True)
            out_html.write_text(html, encoding="utf-8")
            logger.info("Wrote report HTML", extra={"out_html": str(out_html)})
    except Exception as exc:
        _log_exception(
            "report_write_artifacts",
            exc,
            out_json=str(out_json),
            out_md=str(out_md),
            out_html=str(out_html),
        )
        raise

    # Final console feedback
    warning_count = len(getattr(report_obj, "warnings", []) or [])
    if as_json:
        sink.emit_json(payload)
    else:
        sink.emit_kv(
            title="Report Generated",
            items=[
                ("json", str(out_json) if output_format in {"json", "all"} else "(skipped)"),
                ("markdown", str(out_md) if output_format in {"markdown", "all"} else "(skipped)"),
                ("html", str(out_html) if output_format in {"html", "all"} else "(skipped)"),
                ("model", final_model),
                ("profile", report_profile),
                ("warnings", str(warning_count)),
            ],
        )

    logger.info("Report command completed", extra={"warnings": warning_count})
