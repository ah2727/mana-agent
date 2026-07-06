from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from mana_agent.api.services.zip_service import (
    create_zip_from_directory,
    extract_zip_safely,
    require_zip_filename,
)
from mana_agent.commands.chat_analyze_command import run_project_analysis


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_version_string() -> str:
    try:
        import importlib.metadata as md

        return md.version("mana-agent")
    except Exception:
        return "dev"


async def _save_upload(file: UploadFile, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as destination:
        while chunk := await file.read(1024 * 1024):
            destination.write(chunk)


def _find_analysis_root(extraction_dir: Path) -> Path:
    entries = [path for path in extraction_dir.iterdir() if path.name != "__MACOSX"]
    dirs = [path for path in entries if path.is_dir()]
    files = [path for path in entries if path.is_file()]
    if len(dirs) == 1 and not files:
        return dirs[0]
    return extraction_dir


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _clean_items(items: Any, *, limit: int = 12) -> list[Any]:
    if not isinstance(items, list):
        return []
    return items[:limit]


def _project_summary(report: dict[str, Any], context: dict[str, Any], project_name: str) -> str:
    llm = report.get("llm_analysis") if isinstance(report.get("llm_analysis"), dict) else {}
    return str(
        context.get("project_summary")
        or llm.get("project_summary")
        or report.get("project_summary")
        or f"Analysis for {project_name}."
    )


def _analysis_json(project_root: Path, analysis_dir: Path) -> dict[str, Any]:
    report = _load_json(analysis_dir / "report.json")
    context = _load_json(analysis_dir / "agent_context.json")
    inventory = report.get("inventory") if isinstance(report.get("inventory"), dict) else {}
    architecture = report.get("architecture") if isinstance(report.get("architecture"), dict) else {}
    risks = report.get("risks") if isinstance(report.get("risks"), dict) else {}
    recommendations = report.get("recommendations") if isinstance(report.get("recommendations"), dict) else {}
    llm = report.get("llm_analysis") if isinstance(report.get("llm_analysis"), dict) else {}

    important = context.get("important_files") or inventory.get("important_config_files") or inventory.get("entrypoints")
    notes = llm.get("architecture_notes") or architecture.get("area_dependencies") or architecture.get("edges")
    risk_items = context.get("risks") or llm.get("risk_analysis") or risks.get("items")
    suggestions = context.get("recommended_tasks") or llm.get("recommended_tasks") or recommendations.get("items")
    stack = context.get("detected_stack") or inventory.get("detected_frameworks") or []

    return {
        "status": "success",
        "project_name": project_root.name or "project",
        "summary": _project_summary(report, context, project_root.name or "project"),
        "detected_stack": _clean_items(stack),
        "important_files": _clean_items(important),
        "architecture_notes": _clean_items(notes),
        "risks": _clean_items(risk_items),
        "suggestions": _clean_items(suggestions),
        "generated_at": _now_iso(),
    }


def _analysis_markdown(payload: dict[str, Any]) -> str:
    def lines_for(items: list[Any], *, empty: str) -> list[str]:
        if not items:
            return [f"- {empty}"]
        rendered: list[str] = []
        for item in items:
            if isinstance(item, dict):
                label = item.get("file") or item.get("title") or item.get("name") or item.get("message")
                detail = item.get("why") or item.get("summary") or item.get("description") or item.get("severity")
                text = str(label or item)
                if detail:
                    text = f"{text} - {detail}"
                rendered.append(f"- {text}")
            else:
                rendered.append(f"- {item}")
        return rendered

    lines = [
        f"# Mana-Agent Analysis: {payload['project_name']}",
        "",
        "## Project Summary",
        str(payload["summary"]),
        "",
        "## Detected Stack / Languages / Frameworks",
        *lines_for(payload["detected_stack"], empty="No stack signals detected."),
        "",
        "## Important Files / Modules",
        *lines_for(payload["important_files"], empty="No important files detected."),
        "",
        "## Architecture Notes",
        *lines_for(payload["architecture_notes"], empty="No architecture notes detected."),
        "",
        "## Risks / Issues",
        *lines_for(payload["risks"], empty="No high-signal risks detected."),
        "",
        "## Suggested Improvements",
        *lines_for(payload["suggestions"], empty="No suggestions generated."),
        "",
        "## Next Steps",
        "- Review the important files and risks above.",
        "- Add or update focused tests around high-risk modules.",
        "- Run the relevant project checks before making production changes.",
        "",
    ]
    return "\n".join(lines)


def _write_api_outputs(
    *,
    input_filename: str,
    project_root: Path,
    analysis_dir: Path,
    result_dir: Path,
) -> None:
    payload = _analysis_json(project_root, analysis_dir)
    (result_dir / "analysis-report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    (result_dir / "analysis-report.md").write_text(_analysis_markdown(payload), encoding="utf-8")
    manifest = {
        "input_filename": input_filename,
        "extracted_root": project_root.name or "project",
        "analyze_mode": "api_zip",
        "created_at": _now_iso(),
        "mana_agent_version": _safe_version_string(),
        "result_files": [
            "analysis-report.md",
            "analysis-report.json",
        ],
    }
    (result_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


async def run_zip_analysis(*, file: UploadFile, workspace_root: str | Path) -> Path:
    input_filename = require_zip_filename(file.filename)
    workspace = Path(workspace_root)
    upload_path = workspace / "upload" / input_filename
    extraction_dir = workspace / "extracted"
    analysis_dir = workspace / "analysis"
    result_dir = workspace / "result"
    result_zip = workspace / "mana-agent-analysis-result.zip"

    await _save_upload(file, upload_path)
    extract_zip_safely(upload_path, extraction_dir)
    project_root = _find_analysis_root(extraction_dir)

    run_result = run_project_analysis(
        root_dir=project_root,
        output_dir=analysis_dir,
        formats=["json", "markdown"],
    )
    fatal_errors = [error for error in run_result.errors if not str(error).startswith("LLM analysis unavailable:")]
    if fatal_errors:
        raise RuntimeError("; ".join(fatal_errors))

    result_dir.mkdir(parents=True, exist_ok=True)
    _write_api_outputs(
        input_filename=input_filename,
        project_root=project_root,
        analysis_dir=analysis_dir,
        result_dir=result_dir,
    )
    return create_zip_from_directory(
        result_dir,
        result_zip,
        include=["analysis-report.md", "analysis-report.json", "manifest.json"],
    )
