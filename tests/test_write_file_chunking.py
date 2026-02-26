from __future__ import annotations

from pathlib import Path

from mana_analyzer.tools.write_file import build_write_file_tool, safe_finalize_file_parts, safe_write_file_part


def test_safe_write_file_part_then_finalize(tmp_path: Path) -> None:
    part1 = safe_write_file_part(repo_root=tmp_path, path="src/big.txt", content="hello ", part_index=1)
    part2 = safe_write_file_part(repo_root=tmp_path, path="src/big.txt", content="world", part_index=2)

    assert part1["ok"] is True
    assert part2["ok"] is True

    finalize = safe_finalize_file_parts(repo_root=tmp_path, path="src/big.txt")
    assert finalize["ok"] is True
    assert (tmp_path / "src" / "big.txt").read_text(encoding="utf-8") == "hello world"
    assert not (tmp_path / "src" / ".big.txt.parts").exists()


def test_safe_finalize_file_parts_requires_parts(tmp_path: Path) -> None:
    result = safe_finalize_file_parts(repo_root=tmp_path, path="src/missing.txt")
    assert result["ok"] is False
    assert "no parts directory found" in result["error"]


def test_write_file_tool_chunk_then_finalize(tmp_path: Path) -> None:
    tool = build_write_file_tool(repo_root=tmp_path, allowed_prefixes=None)

    r1 = tool.invoke({"path": "docs/out.md", "content": "A", "part_index": 1})
    r2 = tool.invoke({"path": "docs/out.md", "content": "B", "part_index": 2})
    r3 = tool.invoke({"path": "docs/out.md", "finalize": True})

    assert r1["ok"] is True
    assert r2["ok"] is True
    assert r3["ok"] is True
    assert (tmp_path / "docs" / "out.md").read_text(encoding="utf-8") == "AB"

