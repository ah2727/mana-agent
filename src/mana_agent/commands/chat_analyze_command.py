"""Chat ``/analyze`` slash command: analyze the current project and write the
selected report artifacts under ``.mana/``.

This is a thin, read-only wrapper around the existing analysis services
(``DependencyService``, ``StructureService`` and ``AnalyzeService``). It builds
one combined payload and renders the requested formats from it, so adding a new
format never re-runs the analysis. The only side effect is writing artifact
files under the project ``.mana/`` directory.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mana_agent.analysis.models import DependencyGraphReport
from mana_agent.dependencies.dependency_service import DependencyService
from mana_agent.renderers.html_report import render_analyze_html
from mana_agent.services.analyze_service import AnalyzeService
from mana_agent.services.structure_service import StructureService
from mana_agent.analysis.checks import PythonStaticAnalyzer

from .analyze_formats import (
    ANALYZE_ARTIFACTS,
    MENU_FORMATS,
    UnknownAnalyzeFormat,
    parse_analyze_formats,
    parse_menu_choice,
    supported_formats_line,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AnalyzeRunResult",
    "AnalyzeCommandOutcome",
    "is_analyze_command",
    "analyze_command_args",
    "handle_analyze_command",
    "run_project_analysis",
    "prompt_analyze_format_menu",
    "ANALYZE_MENU_TEXT",
]


ANALYZE_MENU_TEXT = (
    "Select output format:\n"
    "\n"
    "1. JSON\n"
    "2. Markdown\n"
    "3. HTML\n"
    "4. DOT graph\n"
    "5. GraphML\n"
    "6. Mermaid diagram\n"
    "7. All formats\n"
    "\n"
    "Enter choice: "
)


@dataclass(slots=True)
class AnalyzeRunResult:
    """Outcome of generating analyze artifacts."""

    output_dir: Path
    formats: list[str]
    written: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AnalyzeCommandOutcome:
    """Result of handling a ``/analyze`` chat command."""

    status: str  # "generated" | "error" | "cancelled"
    message: str
    result: AnalyzeRunResult | None = None


# ---------------------------------------------------------------------------
# Command detection / argument extraction
# ---------------------------------------------------------------------------


def is_analyze_command(question: str) -> bool:
    """True when the chat input is the ``/analyze`` slash command."""
    text = str(question or "").strip()
    if not text.startswith("/analyze"):
        return False
    remainder = text[len("/analyze"):]
    return remainder == "" or remainder[:1].isspace()


def analyze_command_args(question: str) -> str:
    """Return the argument text after ``/analyze`` (may be empty)."""
    text = str(question or "").strip()
    return text[len("/analyze"):].strip()


# ---------------------------------------------------------------------------
# Renderers (one payload -> many formats)
# ---------------------------------------------------------------------------


def _dependency_mermaid(report: DependencyGraphReport) -> str:
    lines = ["graph LR"]
    node_ids: dict[str, str] = {}

    def node_id(name: str) -> str:
        if name not in node_ids:
            node_ids[name] = f"n{len(node_ids)}"
        return node_ids[name]

    edges = list(report.module_edges) + list(report.dependency_edges)
    if not edges:
        lines.append('  empty["No dependency edges detected"]')
        return "\n".join(lines)

    seen: set[tuple[str, str]] = set()
    for edge in edges:
        key = (edge.source, edge.target)
        if key in seen:
            continue
        seen.add(key)
        src_label = str(edge.source).replace('"', "'")
        tgt_label = str(edge.target).replace('"', "'")
        lines.append(
            f'  {node_id(edge.source)}["{src_label}"] --> '
            f'{node_id(edge.target)}["{tgt_label}"]'
        )
    return "\n".join(lines)


def _render_markdown(payload: dict[str, Any]) -> str:
    structure = payload.get("project_structure_analysis") or {}
    summarization = payload.get("summarization") or {}
    dependency = payload.get("dependency_graph") or {}
    findings = payload.get("findings") or []

    root_name = Path(str(payload.get("project_root") or ".")).name or "project"
    lines: list[str] = [f"# Project Analysis: {root_name}", ""]

    lines.append("## Overview")
    lines.append(f"- Project root: {payload.get('project_root', '')}")
    languages = ", ".join(structure.get("language_counts", {}).keys()) or "n/a"
    lines.append(f"- Languages: {languages}")
    frameworks = ", ".join(summarization.get("frameworks", []) or []) or "n/a"
    lines.append(f"- Frameworks: {frameworks}")
    managers = ", ".join(dependency.get("package_managers", []) or []) or "n/a"
    lines.append(f"- Package managers: {managers}")
    lines.append("")

    runtime = dependency.get("runtime_dependencies", []) or []
    dev = dependency.get("dev_dependencies", []) or []
    lines.append("## Dependencies")
    lines.append("")
    lines.append("### Runtime")
    lines.extend([f"- {dep}" for dep in runtime] or ["- (none detected)"])
    lines.append("")
    lines.append("### Dev")
    lines.extend([f"- {dep}" for dep in dev] or ["- (none detected)"])
    lines.append("")

    lines.append("## Findings")
    if findings:
        for item in findings:
            lines.append(
                f"- **{str(item.get('severity', 'info')).upper()}** "
                f"`{item.get('rule_id', '')}` "
                f"{item.get('file_path', '')}:{item.get('line', '?')}"
                f":{item.get('column', '?')} — {item.get('message', '')}"
            )
    else:
        lines.append("No static-analysis findings.")
    lines.append("")

    lines.append("## Dependency Graph")
    lines.append("")
    lines.append("```mermaid")
    lines.append(payload.get("architecture_mermaid", "graph LR"))
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def _render_format(fmt: str, payload: dict[str, Any], dep_report: DependencyGraphReport) -> str:
    if fmt == "json":
        import json

        return json.dumps(payload, indent=2, sort_keys=True, default=str)
    if fmt == "markdown":
        return _render_markdown(payload)
    if fmt == "html":
        return render_analyze_html(payload, _render_markdown(payload))
    if fmt == "dot":
        return dep_report.to_dot()
    if fmt == "graphml":
        return dep_report.to_graphml()
    if fmt == "mermaid":
        return payload.get("architecture_mermaid", "graph LR")
    raise UnknownAnalyzeFormat(fmt)


# ---------------------------------------------------------------------------
# Analysis runner
# ---------------------------------------------------------------------------


def _build_payload(
    root_dir: Path,
    *,
    dep_report: DependencyGraphReport,
) -> dict[str, Any]:
    structure_report = StructureService(include_tests=False).analyze_project(root_dir)

    findings_dicts: list[dict[str, Any]] = []
    try:
        analyze_service = AnalyzeService(analyzer=PythonStaticAnalyzer())
        findings = analyze_service.analyze(root_dir)
        findings_dicts = [item.to_dict() for item in findings]
    except Exception as exc:  # static analysis is best-effort; never block artifacts
        logger.warning("Static analysis skipped for %s: %s", root_dir, exc)

    structure_dict = structure_report.to_dict()
    summarization = {
        "architecture_summary": (
            f"Project at {dep_report.project_root} using "
            f"{', '.join(dep_report.frameworks) or 'no detected frameworks'}."
        ),
        "tech_summary": ", ".join(dep_report.technologies) or "n/a",
        "frameworks": dep_report.frameworks,
        "languages": dep_report.languages,
        "package_managers": dep_report.package_managers,
    }

    return {
        "project_root": dep_report.project_root,
        "findings": findings_dicts,
        "summarization": summarization,
        "tech": {
            "frameworks": dep_report.frameworks,
            "technologies": dep_report.technologies,
            "languages": dep_report.languages,
        },
        "project_structure_analysis": structure_dict,
        "dependency_graph": dep_report.to_dict(),
        "architecture_mermaid": _dependency_mermaid(dep_report),
    }


def run_project_analysis(
    *,
    root_dir: Path | str,
    output_dir: Path | str,
    formats: list[str],
) -> AnalyzeRunResult:
    """Run project analysis once and write the selected artifacts.

    Read-only except for writing files under ``output_dir`` (the ``.mana/``
    directory). Each format is rendered from one shared payload.
    """
    root = Path(root_dir).resolve()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = AnalyzeRunResult(output_dir=out_dir, formats=list(formats))

    dep_report = DependencyService().analyze(root)
    payload = _build_payload(root, dep_report=dep_report)

    for fmt in formats:
        filename = ANALYZE_ARTIFACTS.get(fmt)
        if filename is None:
            result.errors.append(f"Unknown analyze format: {fmt}")
            continue
        try:
            content = _render_format(fmt, payload, dep_report)
            target = out_dir / filename
            target.write_text(content, encoding="utf-8")
            result.written.append(target)
        except Exception as exc:  # pragma: no cover - defensive per-format guard
            logger.warning("Failed to render %s artifact: %s", fmt, exc)
            result.errors.append(f"Failed to generate {fmt}: {exc}")

    return result


# ---------------------------------------------------------------------------
# Interactive menu + command handler
# ---------------------------------------------------------------------------


def prompt_analyze_format_menu(
    *,
    input_func: Callable[[str], str],
) -> list[str]:
    """Show the numbered format menu and return the chosen canonical formats.

    Returns an empty list when the user cancels (blank input). Raises
    ``ValueError`` for invalid menu input (the caller renders the error).
    """
    raw = input_func(ANALYZE_MENU_TEXT)
    return parse_menu_choice(raw)


def _format_summary(result: AnalyzeRunResult) -> str:
    if not result.written:
        return "No analysis artifacts were generated."
    lines = ["Generated analysis artifacts:"]
    for path in result.written:
        try:
            shown = path.relative_to(Path.cwd())
        except ValueError:
            shown = path
        # Prefer the conventional ".mana/<name>" display when possible.
        rel = Path(result.output_dir.name) / path.name
        lines.append(f"- {rel if str(rel).startswith('.mana') else shown}")
    if result.errors:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {err}" for err in result.errors)
    return "\n".join(lines)


def handle_analyze_command(
    args: str,
    *,
    root_dir: Path | str | None,
    output_dir: Path | str | None = None,
    input_func: Callable[[str], str] | None = None,
) -> AnalyzeCommandOutcome:
    """Handle a ``/analyze`` command. Pure of console I/O so it is testable.

    ``args`` is the text after ``/analyze``. ``input_func`` is used to read the
    menu choice when no formats are supplied (defaults to builtin ``input``).
    """
    if root_dir is None:
        return AnalyzeCommandOutcome(
            status="error",
            message=(
                "No root directory is active.\n\n"
                "Start chat with:\n"
                "mana-agent chat --root-dir /path/to/project"
            ),
        )

    root = Path(root_dir).resolve()
    out_dir = Path(output_dir) if output_dir is not None else (root / ".mana")

    try:
        formats = parse_analyze_formats(args)
    except UnknownAnalyzeFormat as exc:
        return AnalyzeCommandOutcome(
            status="error",
            message=(
                f"Unknown analyze format: {exc.token}\n\n"
                f"Supported formats:\n{supported_formats_line()}"
            ),
        )

    if not formats:
        reader = input_func or (lambda prompt: input(prompt))
        try:
            formats = prompt_analyze_format_menu(input_func=reader)
        except (ValueError, EOFError, KeyboardInterrupt) as exc:
            return AnalyzeCommandOutcome(
                status="error",
                message=(
                    f"{exc}\n\nValid choices are 1-7 "
                    f"(e.g. 1, or 1,2,3, or 7 for all)."
                ),
            )
        if not formats:
            return AnalyzeCommandOutcome(
                status="cancelled",
                message="Analyze cancelled — no format selected.",
            )

    result = run_project_analysis(root_dir=root, output_dir=out_dir, formats=formats)
    return AnalyzeCommandOutcome(
        status="generated",
        message=_format_summary(result),
        result=result,
    )


# Keep a reference so linters do not flag the menu-format list as unused; it is
# part of the public surface used by tests and docs.
_MENU_FORMATS = MENU_FORMATS
