from __future__ import annotations

from pathlib import Path
from mana_agent.workspaces.paths import repository_analysis_dir, repository_id_for_path

import pytest

from mana_agent.commands.analyze_formats import (
    ANALYZE_ARTIFACTS,
    UnknownAnalyzeFormat,
    parse_analyze_formats,
    parse_menu_choice,
)
from mana_agent.commands.chat_analyze_command import (
    analyze_command_args,
    handle_analyze_command,
    is_analyze_command,
    run_project_analysis,
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args,expected",
    [
        ("json", ["json"]),
        ("markdown", ["markdown"]),
        ("md", ["markdown"]),
        ("html", ["html"]),
        ("dot", ["dot"]),
        ("graphml", ["graphml"]),
        ("mermaid", ["mermaid"]),
        ("all", list(ANALYZE_ARTIFACTS.keys())),
        ("json markdown html", ["json", "markdown", "html"]),
        ("--format json,markdown,html", ["json", "markdown", "html"]),
        ("json, md , html", ["json", "markdown", "html"]),
        ("", []),
        (None, []),
    ],
)
def test_parse_analyze_formats(args, expected) -> None:
    assert parse_analyze_formats(args) == expected


def test_parse_analyze_formats_dedupes() -> None:
    assert parse_analyze_formats("json json md markdown") == ["json", "markdown"]


def test_parse_analyze_formats_unknown_raises() -> None:
    with pytest.raises(UnknownAnalyzeFormat) as exc:
        parse_analyze_formats("pdf")
    assert exc.value.token == "pdf"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1", ["json"]),
        ("2", ["markdown"]),
        ("3", ["html"]),
        ("1,2,3", ["json", "markdown", "html"]),
        ("1 2 3", ["json", "markdown", "html"]),
        ("7", list(ANALYZE_ARTIFACTS.keys())),
        ("", []),
    ],
)
def test_parse_menu_choice(raw, expected) -> None:
    assert parse_menu_choice(raw) == expected


@pytest.mark.parametrize("raw", ["8", "0", "abc", "1,9"])
def test_parse_menu_choice_invalid(raw) -> None:
    with pytest.raises(ValueError):
        parse_menu_choice(raw)


# ---------------------------------------------------------------------------
# Command detection
# ---------------------------------------------------------------------------


def test_is_analyze_command() -> None:
    assert is_analyze_command("/analyze")
    assert is_analyze_command("/analyze all")
    assert is_analyze_command("  /analyze json  ")
    assert not is_analyze_command("/analyzed")
    assert not is_analyze_command("analyze the project")
    assert not is_analyze_command("/flow show")


def test_analyze_command_args() -> None:
    assert analyze_command_args("/analyze") == ""
    assert analyze_command_args("/analyze all") == "all"
    assert analyze_command_args("/analyze --format json,md") == "--format json,md"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_project(tmp_path: Path) -> Path:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "a.py").write_text("import os\nfrom pkg import b\n\n\ndef run():\n    return b.x\n")
    (pkg / "b.py").write_text("x = 1\n")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\ndependencies = ['requests']\n"
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Artifact generation (direct mode)
# ---------------------------------------------------------------------------


def test_run_project_analysis_writes_selected(sample_project: Path) -> None:
    out_dir = sample_project / ".mana" / "analyze"
    result = run_project_analysis(
        root_dir=sample_project,
        output_dir=out_dir,
        formats=["json", "markdown", "html"],
    )
    names = {p.name for p in result.written}
    assert {"report.json", "report.md", "agent_context.json", "inventory.json", "symbols.json", "dependencies.json", "risks.json"} <= names
    assert {"audit_report.json", "audit_report.md", "audit_report.html"} <= names
    assert "analyze.html" in names
    for path in result.written:
        assert path.exists() and path.read_text(encoding="utf-8").strip()


def test_run_project_analysis_all_formats(sample_project: Path) -> None:
    out_dir = sample_project / ".mana"
    result = run_project_analysis(
        root_dir=sample_project,
        output_dir=out_dir,
        formats=list(ANALYZE_ARTIFACTS.keys()),
    )
    names = {p.name for p in result.written}
    assert {"report.json", "report.md", "agent_context.json", "analyze.html", "analyze.dot", "analyze.graphml", "diagram.mmd"} <= names
    assert {"audit_report.json", "audit_report.md", "audit_report.html"} <= names
    assert (out_dir / "diagram.mmd").read_text(encoding="utf-8").startswith("graph LR")
    assert "digraph" in (out_dir / "analyze.dot").read_text(encoding="utf-8")
    assert "graphml" in (out_dir / "analyze.graphml").read_text(encoding="utf-8")


def test_handle_analyze_direct_all(sample_project: Path) -> None:
    outcome = handle_analyze_command("all", root_dir=sample_project)
    assert outcome.status == "generated"
    assert outcome.result is not None
    assert len(outcome.result.written) >= len(ANALYZE_ARTIFACTS)
    # The modern path emits the rich .mana/analyze report set and a compact,
    # context-backed chat summary.
    assert "Analysis completed." in outcome.message
    assert "audit_report.md" in outcome.message
    assert "Summary:" in outcome.message


def test_handle_analyze_direct_subset(sample_project: Path) -> None:
    outcome = handle_analyze_command("json markdown", root_dir=sample_project)
    assert outcome.status == "generated"
    names = {p.name for p in outcome.result.written}
    assert {"report.json", "report.md", "agent_context.json"} <= names
    assert {"audit_report.json", "audit_report.md", "audit_report.html"} <= names


# ---------------------------------------------------------------------------
# Menu mode
# ---------------------------------------------------------------------------


def test_handle_analyze_menu_choice_one_creates_json(sample_project: Path) -> None:
    outcome = handle_analyze_command("", root_dir=sample_project, input_func=lambda _p: "1")
    assert outcome.status == "generated"
    assert "report.json" in {p.name for p in outcome.result.written}


def test_handle_analyze_menu_choice_seven_creates_all(sample_project: Path) -> None:
    outcome = handle_analyze_command("", root_dir=sample_project, input_func=lambda _p: "7")
    assert outcome.status == "generated"
    assert {"report.json", "report.md", "agent_context.json"} <= {p.name for p in outcome.result.written}


def test_handle_analyze_menu_blank_cancels(sample_project: Path) -> None:
    outcome = handle_analyze_command("", root_dir=sample_project, input_func=lambda _p: "")
    assert outcome.status == "generated"
    assert (repository_analysis_dir(repository_id_for_path(sample_project)) / "report.md").exists()


def test_handle_analyze_menu_invalid_does_not_crash(sample_project: Path) -> None:
    outcome = handle_analyze_command("", root_dir=sample_project, input_func=lambda _p: "99")
    assert outcome.status == "generated"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_handle_analyze_invalid_format(sample_project: Path) -> None:
    outcome = handle_analyze_command("pdf", root_dir=sample_project)
    assert outcome.status == "error"
    assert "Unknown analyze format: pdf" in outcome.message
    assert "Supported formats" in outcome.message


def test_handle_analyze_no_root() -> None:
    outcome = handle_analyze_command("json", root_dir=None)
    assert outcome.status == "error"
    assert "No root directory is active" in outcome.message


# ---------------------------------------------------------------------------
# Read-only guarantee (only .mana/ is written)
# ---------------------------------------------------------------------------


def test_analyze_does_not_modify_source_files(sample_project: Path) -> None:
    before = {
        p: p.read_text(encoding="utf-8")
        for p in sample_project.rglob("*")
        if p.is_file() and ".mana" not in p.parts
    }
    handle_analyze_command("all", root_dir=sample_project)
    after = {
        p: p.read_text(encoding="utf-8")
        for p in sample_project.rglob("*")
        if p.is_file() and ".mana" not in p.parts
    }
    assert before == after
