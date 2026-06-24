from __future__ import annotations

import logging
from pathlib import Path

from mana_agent.analysis.models import SearchHit
from mana_agent.services.ask_service import (
    SEMANTIC_INDEX_MISSING_WARNING,
    AskService,
)
from mana_agent.vector_store.faiss_store import FaissStore


class _EmptyStore:
    """Store that finds nothing (simulates missing/empty FAISS index)."""

    def search(self, _index_dir: Path, query: str, k: int) -> list[SearchHit]:
        return []


class _HitStore:
    def __init__(self, hits: list[SearchHit]) -> None:
        self._hits = hits

    def search(self, _index_dir: Path, query: str, k: int) -> list[SearchHit]:
        return self._hits


class _FakeQnA:
    def __init__(self) -> None:
        self.last_context: str | None = None

    def run(self, question: str, context: str) -> str:
        self.last_context = context
        return "synthesized answer"


def _make_project(tmp_path: Path) -> Path:
    (tmp_path / "mod.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    return tmp_path


def test_ask_falls_back_to_project_search_when_index_missing(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    qna = _FakeQnA()
    service = AskService(store=_EmptyStore(), qna_chain=qna, project_root=project)

    response = service.ask(index_dir=tmp_path / ".mana" / "index", question="add", k=5)

    assert SEMANTIC_INDEX_MISSING_WARNING in response.warnings
    assert response.sources, "fallback should surface matched files/lines as sources"
    assert any("mod.py" in src.file_path for src in response.sources)
    # The fallback ran the QnA chain over the project-search context.
    assert qna.last_context is not None and "mod.py" in qna.last_context
    assert "mana-agent index" in response.answer


def test_ask_does_not_emit_dead_end_message(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    service = AskService(store=_EmptyStore(), qna_chain=_FakeQnA(), project_root=project)
    response = service.ask(index_dir=tmp_path, question="nonexistent_symbol_xyz", k=3)
    # Even with no matches, we must not return the old dead-end message.
    assert "could not find relevant indexed code context" not in response.answer.lower()
    assert "Semantic index is missing" not in response.answer
    assert "No direct project matches" in response.answer
    assert SEMANTIC_INDEX_MISSING_WARNING in response.warnings


def test_ask_command_inventory_fallback_lists_cli_commands(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "demo"',
                "",
                "[project.scripts]",
                'demo-cli = "demo.commands:app"',
            ]
        ),
        encoding="utf-8",
    )
    package = tmp_path / "demo"
    package.mkdir()
    (package / "commands.py").write_text(
        "\n".join(
            [
                "import typer",
                "",
                "app = typer.Typer()",
                "",
                "@app.command()",
                "def run():",
                "    pass",
            ]
        ),
        encoding="utf-8",
    )
    service = AskService(store=_EmptyStore(), qna_chain=_FakeQnA(), project_root=tmp_path)

    response = service.ask(index_dir=tmp_path / ".mana" / "index", question="give me all command of this project", k=3)

    assert response.warnings == []
    assert "Semantic index is missing" not in response.answer
    assert "No direct matches" not in response.answer
    assert "`demo-cli` console script" in response.answer
    assert "`demo-cli run`" in response.answer
    assert response.sources


def test_ask_command_inventory_fallback_handles_exist_agent_wording(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "demo"',
                "",
                "[project.scripts]",
                'demo-cli = "demo.commands:app"',
            ]
        ),
        encoding="utf-8",
    )
    package = tmp_path / "demo"
    package.mkdir()
    (package / "commands.py").write_text(
        "\n".join(
            [
                "import typer",
                "",
                "app = typer.Typer()",
                "",
                "@app.command()",
                "def run():",
                "    pass",
            ]
        ),
        encoding="utf-8",
    )
    service = AskService(store=_EmptyStore(), qna_chain=_FakeQnA(), project_root=tmp_path)

    response = service.ask(index_dir=tmp_path / ".mana" / "index", question="command exist in this agent", k=3)

    assert response.warnings == []
    assert "Semantic index is missing" not in response.answer
    assert "`demo-cli` console script" in response.answer
    assert "`demo-cli run`" in response.answer


def test_ask_missing_faiss_index_uses_quiet_fallback(tmp_path: Path, caplog) -> None:
    service = AskService(store=FaissStore(embeddings=object()), qna_chain=_FakeQnA(), project_root=tmp_path)

    with caplog.at_level(logging.WARNING):
        response = service.ask(index_dir=tmp_path / ".mana" / "index", question="nonexistent_symbol_xyz", k=3)

    assert SEMANTIC_INDEX_MISSING_WARNING in response.warnings
    assert "Cannot search: FAISS index not found" not in caplog.text


def test_ask_normal_faiss_path_still_works(tmp_path: Path) -> None:
    hits = [
        SearchHit(
            score=0.9,
            file_path="/tmp/proj/app.py",
            start_line=1,
            end_line=3,
            symbol_name="add",
            snippet="def add(a, b): return a + b",
        )
    ]
    qna = _FakeQnA()
    service = AskService(store=_HitStore(hits), qna_chain=qna, project_root=tmp_path)
    response = service.ask(index_dir=tmp_path, question="how does add work?", k=5)

    assert response.sources == hits
    assert response.answer == "synthesized answer"
    assert SEMANTIC_INDEX_MISSING_WARNING not in response.warnings
    # Normal path uses the semantic-context renderer.
    assert qna.last_context is not None and "symbol:" in qna.last_context
