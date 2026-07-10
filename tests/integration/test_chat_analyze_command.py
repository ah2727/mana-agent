from __future__ import annotations

from pathlib import Path
from mana_agent.workspaces.paths import repository_analysis_dir, repository_id_for_path


def _analysis(project: Path) -> Path:
    return repository_analysis_dir(repository_id_for_path(project))

import pytest

from mana_agent.commands.chat_analyze_command import (
    ANALYZE_MENU_TEXT,
    handle_analyze_command,
)


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    pkg = tmp_path / "app"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text("from app import util\n\n\ndef main():\n    return util.value\n")
    (pkg / "util.py").write_text("value = 42\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'app'\n")
    return tmp_path


def test_menu_text_lists_all_seven_options() -> None:
    for label in ["1. JSON", "2. Markdown", "3. HTML", "4. DOT", "5. GraphML", "6. Mermaid", "7. All"]:
        assert label in ANALYZE_MENU_TEXT


def test_empty_analyze_opens_menu_and_uses_input(project: Path) -> None:
    seen_prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        seen_prompts.append(prompt)
        return "1"

    outcome = handle_analyze_command("", root_dir=project, input_func=fake_input)
    assert seen_prompts == []
    assert outcome.status == "generated"
    assert (_analysis(project) / "report.json").exists()


def test_menu_choice_one_creates_json(project: Path) -> None:
    outcome = handle_analyze_command("", root_dir=project, input_func=lambda _p: "1")
    assert outcome.status == "generated"
    assert (_analysis(project) / "report.json").exists()
    assert (_analysis(project) / "report.md").exists()


def test_menu_choice_two_creates_markdown(project: Path) -> None:
    handle_analyze_command("", root_dir=project, input_func=lambda _p: "2")
    assert (_analysis(project) / "report.md").exists()


def test_menu_choice_three_creates_html(project: Path) -> None:
    handle_analyze_command("html", root_dir=project)
    assert (_analysis(project) / "analyze.html").exists()


def test_menu_choice_all_creates_every_artifact(project: Path) -> None:
    handle_analyze_command("all", root_dir=project)
    mana = _analysis(project)
    for name in [
        "report.json",
        "report.md",
        "agent_context.json",
        "analyze.html",
        "analyze.dot",
        "analyze.graphml",
        "diagram.mmd",
    ]:
        assert (mana / name).exists(), name


def test_direct_form_all(project: Path) -> None:
    outcome = handle_analyze_command("all", root_dir=project)
    assert outcome.status == "generated"
    assert (_analysis(project) / "diagram.mmd").exists()


def test_direct_form_format_flag(project: Path) -> None:
    outcome = handle_analyze_command("--format json,markdown,html", root_dir=project)
    assert outcome.status == "generated"
    names = {p.name for p in outcome.result.written}
    assert {"report.json", "report.md", "agent_context.json", "analyze.html"} <= names


def test_invalid_format_clean_error(project: Path) -> None:
    outcome = handle_analyze_command("pdf", root_dir=project)
    assert outcome.status == "error"
    assert "Unknown analyze format" in outcome.message
    assert not (project / ".mana").exists()
