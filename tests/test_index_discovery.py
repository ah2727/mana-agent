from pathlib import Path

from mana_analyzer.utils.index_discovery import discover_index_dirs


def test_discover_index_dirs_finds_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "b" / ".mana/index").mkdir(parents=True)
    (tmp_path / "a" / ".mana/index").mkdir(parents=True)

    result = discover_index_dirs(tmp_path)
    assert result == [
        (tmp_path / "a" / ".mana/index").resolve(),
        (tmp_path / "b" / ".mana/index").resolve(),
    ]


def test_discover_index_dirs_skips_excluded_dirs(tmp_path: Path) -> None:
    (tmp_path / "node_modules" / "x" / ".mana/index").mkdir(parents=True)
    (tmp_path / "pkg" / ".mana/index").mkdir(parents=True)

    result = discover_index_dirs(tmp_path)
    assert result == [(tmp_path / "pkg" / ".mana/index").resolve()]
