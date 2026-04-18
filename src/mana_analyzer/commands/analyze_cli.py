from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv

load_dotenv()

from .cli_internal import *
from .output import build_output_sink
from mana_analyzer.renderers.html_report import render_analyze_html
from mana_analyzer.utils.guards import guard_root


def _read_source(path: str) -> str:
    p = Path(path)
    if p.is_file():
        try:
            return p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
    return ""


def _count_findings_by_severity(findings: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in findings:
        sev = str(getattr(finding, "severity", "unknown") or "unknown").lower()
        counts[sev] = counts.get(sev, 0) + 1
    return counts


def _parse_analysis_lines(raw_text: str) -> list[str]:
    text = (raw_text or "").strip()
    if not text:
        return []

    # Prefer strict JSON payloads from the LLM for deterministic parsing.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            raw_lines = parsed.get("analysis_lines", [])
            if isinstance(raw_lines, list):
                return [str(item).strip() for item in raw_lines if str(item).strip()]
    except Exception:
        pass

    lines: list[str] = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-*").strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def _build_project_structure_analysis(
    *,
    settings: Settings,
    model: str,
    deps: Any,
    structure_report: Any,
    summary_dict: dict[str, Any],
    findings: list[Any],
    min_lines: int = 500,
    max_lines: int = 1000,
) -> dict[str, Any]:
    if min_lines < 1:
        min_lines = 1
    if max_lines < min_lines:
        max_lines = min_lines

    target_lines = min(max((min_lines + max_lines) // 2, min_lines), max_lines)

    dep_payload = deps.to_dict() if hasattr(deps, "to_dict") else {}
    struct_payload = structure_report.to_dict() if hasattr(structure_report, "to_dict") else {}

    files = summary_dict.get("files", []) if isinstance(summary_dict, dict) else []
    sampled_files = files[:12] if isinstance(files, list) else []

    context_payload = {
        "dependency_report": {
            "languages": dep_payload.get("languages", []),
            "frameworks": dep_payload.get("frameworks", []),
            "technologies": dep_payload.get("technologies", []),
            "runtime_dependencies": dep_payload.get("runtime_dependencies", [])[:80],
            "dev_dependencies": dep_payload.get("dev_dependencies", [])[:80],
            "module_edge_count": len(dep_payload.get("module_edges", []) or []),
            "dependency_edge_count": len(dep_payload.get("dependency_edges", []) or []),
            "manifests": dep_payload.get("manifests", [])[:50],
        },
        "structure_report": {
            "project_root": struct_payload.get("project_root", ""),
            "language_counts": struct_payload.get("language_counts", {}),
            "file_counts": struct_payload.get("file_counts", {}),
            "commands": (struct_payload.get("commands", []) or [])[:50],
            "directories": (struct_payload.get("directories", []) or [])[:120],
            "discovery_stats": struct_payload.get("discovery_stats", {}),
        },
        "findings_summary": {
            "total": len(findings),
            "by_severity": _count_findings_by_severity(findings),
        },
        "sampled_file_summaries": sampled_files,
    }

    system_prompt = (
        "You are a principal software architecture reviewer. "
        "Return strict JSON only. Do not use markdown."
    )

    lines: list[str] = []
    warnings: list[str] = []

    try:
        llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=model,
            base_url=settings.openai_base_url,
            temperature=0,
        )

        for _ in range(3):
            remaining = target_lines - len(lines)
            if remaining <= 0:
                break

            user_prompt = (
                "Generate project structure analysis lines for this repository context. "
                f"Return exactly {remaining} unique lines as a JSON object with key `analysis_lines`. "
                "Each line must be a single sentence, technical, evidence-grounded, and actionable. "
                "No numbering, no bullets, no markdown.\n\n"
                f"Context JSON:\n{json.dumps(context_payload, ensure_ascii=False)}\n\n"
                f"Existing lines to avoid repeating:\n{json.dumps(lines, ensure_ascii=False)}"
            )

            response = llm.invoke([
                ("system", system_prompt),
                ("human", user_prompt),
            ])
            parsed = _parse_analysis_lines(str(getattr(response, "content", "")))
            if not parsed:
                break

            seen = set(lines)
            for line in parsed:
                if line not in seen:
                    lines.append(line)
                    seen.add(line)
                if len(lines) >= target_lines:
                    break

    except Exception as exc:
        warnings.append(f"project_structure_llm_failed: {type(exc).__name__}: {exc}")

    generated_by = "llm"

    if len(lines) < min_lines:
        generated_by = "llm+deterministic-fallback"
        for idx in range(len(lines) + 1, min_lines + 1):
            lines.append(
                f"Supplemental structure insight {idx}: maintain explicit boundaries between parsing, analysis, orchestration, and tooling modules to keep change impact localized."
            )

    if len(lines) > max_lines:
        lines = lines[:max_lines]

    markdown_lines = ["# Project Structure Analysis", ""]
    markdown_lines.extend([f"{i:03d}. {line}" for i, line in enumerate(lines, start=1)])

    return {
        "title": "Project Structure Analysis",
        "generated_by": generated_by,
        "line_count": len(lines),
        "analysis_lines": [
            {"line_number": i, "text": line}
            for i, line in enumerate(lines, start=1)
        ],
        "markdown": "\n".join(markdown_lines) + "\n",
        "warnings": warnings,
    }


@app.command()
@guard_root
def analyze(
    path: str,
    fail_on: str = typer.Option("none", "--fail-on"),
    model: str | None = typer.Option(None, "--model"),
    llm_max_files: int = typer.Option(10, "--llm-max-files"),
    include_tests: bool = typer.Option(True, "--include-tests"),
    output_format: str = typer.Option("all", "--output-format"),
    chain_profile: str = typer.Option("default", "--chain-profile"),
    chain_config: str | None = typer.Option(None, "--chain-config"),
    structure_min_lines: int = typer.Option(500, "--structure-min-lines"),
    structure_max_lines: int = typer.Option(1000, "--structure-max-lines"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:

    final_model = model or os.environ.get("LLM_MODEL")
    if not final_model:
        raise typer.BadParameter(
            "No LLM model specified. "
            "Pass --model or set LLM_MODEL."
        )

    if output_format not in {"json", "markdown", "html", "all"}:
        raise typer.BadParameter("--output-format must be one of: json, markdown, html, all")

    if structure_min_lines < 1:
        raise typer.BadParameter("--structure-min-lines must be >= 1")
    if structure_max_lines < structure_min_lines:
        raise typer.BadParameter("--structure-max-lines must be >= --structure-min-lines")

    root = Path(path).resolve()
    output_file = _resolve_output_file(output_dir=root)
    sink = build_output_sink(command_name="analyze", json_mode=as_json, output_file=output_file, console=console)

    logger.info("Analyze started", extra={"path": path, "model": final_model})
    settings = Settings()

    # ---------------------------------------------------
    # 1) Dependency scan (single source of truth)
    # ---------------------------------------------------
    try:
        dependency_service = build_dependency_service()
        deps = dependency_service.analyze(root)
        logger.info("Dependency scan found %d files", len(deps.files))
    except Exception as exc:
        _log_exception("deps", exc, path=path)
        raise

    # ---------------------------------------------------
    # 2) Static analysis
    # ---------------------------------------------------
    try:
        static_service = build_analyze_service()
        static_findings = static_service.analyze(path)
    except Exception as exc:
        _log_exception("static_analyze", exc, path=path)
        raise

    # ---------------------------------------------------
    # 3) LLM file-level analysis
    # ---------------------------------------------------
    try:
        analyze_chain, _file_agent = build_llm_analyze_service(
            settings,
            model_override=final_model,
        )

        llm_findings = []
        files = deps.files[:llm_max_files]

        for file_path in files:
            try:
                source = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if not source:
                continue

            try:
                file_findings = analyze_chain.run(
                    file_path=str(file_path.relative_to(root)),
                    source=source[:12000],
                    static_findings=static_findings,
                )
                llm_findings.extend(file_findings)
            except Exception as exc:
                logger.warning("LLM analyze failed for %s: %s", file_path, exc)

    except Exception as exc:
        _log_exception("llm_analyze", exc, path=path)
        raise

    findings = static_findings + llm_findings
    findings = sorted(
        {
            (f.rule_id, f.severity, f.file_path, f.line, f.column): f
            for f in findings
        }.values(),
        key=lambda f: (f.file_path, f.line, f.column, f.rule_id),
    )

    # ---------------------------------------------------
    # 4) Emit findings (console)
    # ---------------------------------------------------
    if as_json:
        sink.emit_json([f.to_dict() for f in findings])
    else:
        if not findings:
            sink.emit_success("No findings.")
        else:
            rows = [
                [
                    f.severity,
                    f.rule_id,
                    f"{f.file_path}:{f.line}:{f.column}",
                    f.message,
                ]
                for f in findings
            ]
            sink.emit_table(
                title=f"Findings ({len(findings)})",
                columns=["severity", "rule", "location", "message"],
                rows=rows,
            )

    # ---------------------------------------------------
    # 5) Structure analysis
    # ---------------------------------------------------
    structure_service = StructureService(include_tests=include_tests)
    structure_report = structure_service.analyze_project(path)

    # ---------------------------------------------------
    # 6) Repository describe (file summaries)
    # ---------------------------------------------------
    try:
        describe_service = build_describe_service()
        summary_dict = describe_service.describe(root)
    except Exception as exc:
        _log_exception("describe", exc, path=path)
        raise

    # ---------------------------------------------------
    # 7) Tech summary
    # ---------------------------------------------------
    languages = sorted(set(getattr(deps, "languages", [])))
    tech_payload = {
        "languages": languages,
        "file_count": len(deps.files),
        "chain_profile": chain_profile,
        "chain_config": chain_config or "",
    }

    # ---------------------------------------------------
    # 8) LLM Project Structure Analysis (500-1000 lines)
    # ---------------------------------------------------
    project_structure_analysis = _build_project_structure_analysis(
        settings=settings,
        model=final_model,
        deps=deps,
        structure_report=structure_report,
        summary_dict=summary_dict,
        findings=findings,
        min_lines=structure_min_lines,
        max_lines=structure_max_lines,
    )

    # ---------------------------------------------------
    # 9) Final artifacts
    # ---------------------------------------------------
    payload = structure_report.to_dict()
    payload["findings"] = [f.to_dict() for f in findings]
    payload["summarization"] = summary_dict
    payload["tech"] = tech_payload
    payload["project_structure_analysis"] = {
        "title": project_structure_analysis["title"],
        "generated_by": project_structure_analysis["generated_by"],
        "line_count": project_structure_analysis["line_count"],
        "analysis_lines": project_structure_analysis["analysis_lines"],
        "warnings": project_structure_analysis["warnings"],
    }

    markdown = (
        _render_findings_markdown(findings)
        + "\n\n"
        + structure_service.render_markdown(structure_report)
        + "\n\n"
        + _render_repository_summary_markdown(summary_dict)
        + "\n\n"
        + project_structure_analysis["markdown"]
    )

    html = render_analyze_html(payload, markdown)

    out_json, out_md, out_html = _resolve_analyze_artifact_paths(path)

    if output_format in {"json", "all"}:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if output_format in {"markdown", "all"}:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(markdown, encoding="utf-8")

    if output_format in {"html", "all"}:
        out_html.parent.mkdir(parents=True, exist_ok=True)
        out_html.write_text(html, encoding="utf-8")

    # ---------------------------------------------------
    # 10) fail-on logic
    # ---------------------------------------------------
    if fail_on == "warning" and any(
        f.severity in {"warning", "error"} for f in findings
    ):
        raise typer.Exit(code=1)

    if fail_on == "error" and any(
        f.severity == "error" for f in findings
    ):
        raise typer.Exit(code=1)
