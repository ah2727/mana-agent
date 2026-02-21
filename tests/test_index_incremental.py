from __future__ import annotations

from pathlib import Path

from mana_analyzer.analysis.chunker import CodeChunker
from mana_analyzer.analysis.models import CodeChunk
from mana_analyzer.parsers.multi_parser import MultiLanguageParser
from mana_analyzer.services.index_service import IndexService


class FakeStore:
    def __init__(self) -> None:
        self.calls: list[tuple[list[CodeChunk], list[str]]] = []

    def upsert_chunks(self, _index_dir: Path, chunks: list[CodeChunk], delete_ids: list[str]) -> None:
        self.calls.append((chunks, delete_ids))


def test_incremental_index_updates_only_changed_files(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    file_a = project / "a.py"
    file_b = project / "b.py"
    file_a.write_text('"""A"""\n\ndef one():\n    """one"""\n    return 1\n', encoding="utf-8")
    file_b.write_text('"""B"""\n\ndef two():\n    """two"""\n    return 2\n', encoding="utf-8")

    index_dir = tmp_path / "index"
    store = FakeStore()
    service = IndexService(MultiLanguageParser(), CodeChunker(), store)

    first = service.index(project, index_dir, rebuild=False)
    assert first["indexed_files"] == 2

    file_a.write_text('"""A"""\n\ndef one():\n    """changed"""\n    return 10\n', encoding="utf-8")
    second = service.index(project, index_dir, rebuild=False)

    assert second["indexed_files"] == 1
    assert second["removed_chunks"] >= 1


def test_incremental_index_includes_non_python_source_files(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    py_file = project / "a.py"
    ts_file = project / "b.ts"
    dart_file = project / "c.dart"
    py_file.write_text("def a():\n    return 1\n", encoding="utf-8")
    ts_file.write_text("export function b() { return 2; }\n", encoding="utf-8")
    dart_file.write_text("String c() { return '3'; }\n", encoding="utf-8")

    index_dir = tmp_path / "index"
    store = FakeStore()
    service = IndexService(MultiLanguageParser(), CodeChunker(), store)

    result = service.index(project, index_dir, rebuild=False)
    assert result["indexed_files"] == 3
