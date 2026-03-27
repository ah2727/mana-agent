from __future__ import annotations

from pathlib import Path

from mana_analyzer.tools.apply_patch import build_apply_patch_tool
from mana_analyzer.tools.write_file import build_write_file_tool


def test_apply_patch_tool_accepts_patch_alias(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_safe_apply_patch(
        *,
        repo_root: Path,
        patch: str,
        allowed_prefixes,
        check_only: bool,
        **_kwargs: object,
    ) -> dict:
        captured["repo_root"] = repo_root
        captured["patch"] = patch
        captured["allowed_prefixes"] = allowed_prefixes
        captured["check_only"] = check_only
        return {"ok": True, "touched_files": ["sample.py"], "check_only": check_only}

    monkeypatch.setattr("mana_analyzer.tools.apply_patch.safe_apply_patch", _fake_safe_apply_patch)

    tool = build_apply_patch_tool(repo_root=tmp_path, allowed_prefixes=None)
    result = tool.invoke({"patch": "*** Begin Patch\n*** End Patch", "check_only": True})

    assert result["ok"] is True
    assert captured["patch"] == "*** Begin Patch\n*** End Patch"
    assert captured["check_only"] is True


def test_write_file_tool_accepts_text_alias(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_safe_write_file(
        *,
        repo_root: Path,
        path: str,
        content: str,
        allowed_prefixes,
    ) -> dict:
        captured["repo_root"] = repo_root
        captured["path"] = path
        captured["content"] = content
        captured["allowed_prefixes"] = allowed_prefixes
        return {"ok": True, "path": path, "bytes_written": len(content.encode("utf-8")), "sha256": "", "error": ""}

    monkeypatch.setattr("mana_analyzer.tools.write_file.safe_write_file", _fake_safe_write_file)

    tool = build_write_file_tool(repo_root=tmp_path, allowed_prefixes=None)
    result = tool.invoke({"path": "src/new_file.py", "text": "print('ok')\n"})

    assert result["ok"] is True
    assert captured["path"] == "src/new_file.py"
    assert captured["content"] == "print('ok')\n"
