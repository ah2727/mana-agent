from __future__ import annotations

import subprocess
from pathlib import Path

from mana_analyzer.tools import apply_patch as apply_patch_mod
from mana_analyzer.tools.apply_patch import safe_apply_patch


def _force_git_failure(monkeypatch) -> None:
    def _run_fail(cmd: list[str], *, cwd: Path, stdin_text: str) -> subprocess.CompletedProcess[str]:
        _ = (cwd, stdin_text)
        return subprocess.CompletedProcess(cmd, 1, "", "forced git failure")

    monkeypatch.setattr(apply_patch_mod, "_run", _run_fail)


def _set_which(monkeypatch, *, git: bool, perl: bool) -> None:
    def _which(name: str) -> str | None:
        if name == "git":
            return "/usr/bin/git" if git else None
        if name == "perl":
            return "/usr/bin/perl" if perl else None
        return None

    monkeypatch.setattr(apply_patch_mod.shutil, "which", _which)


def test_check_only_full_chain_falls_back_to_python_without_mutation(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "src" / "demo.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("old\nkeep\n", encoding="utf-8")

    diff = (
        "diff --git a/src/demo.txt b/src/demo.txt\n"
        "--- a/src/demo.txt\n"
        "+++ b/src/demo.txt\n"
        "@@ -1,2 +1,2 @@\n"
        "-old\n"
        "+new\n"
        " keep\n"
    )

    _set_which(monkeypatch, git=True, perl=False)
    _force_git_failure(monkeypatch)

    result = safe_apply_patch(repo_root=tmp_path, diff=diff, check_only=True)

    assert result["ok"] is True
    assert result["check_only"] is True
    assert result["strategy"] == "python"
    assert target.read_text(encoding="utf-8") == "old\nkeep\n"


def test_apply_full_chain_uses_write_file_after_python_compute(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "src" / "demo.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("old\nkeep\n", encoding="utf-8")

    diff = (
        "diff --git a/src/demo.txt b/src/demo.txt\n"
        "--- a/src/demo.txt\n"
        "+++ b/src/demo.txt\n"
        "@@ -1,2 +1,2 @@\n"
        "-old\n"
        "+new\n"
        " keep\n"
    )

    _set_which(monkeypatch, git=True, perl=False)
    _force_git_failure(monkeypatch)

    result = safe_apply_patch(repo_root=tmp_path, diff=diff, check_only=False)

    assert result["ok"] is True
    assert result["strategy"] == "write_file"
    assert target.read_text(encoding="utf-8") == "new\nkeep\n"


def test_write_file_fallback_rolls_back_all_files_on_partial_failure(tmp_path: Path, monkeypatch) -> None:
    a_file = tmp_path / "src" / "a.txt"
    b_file = tmp_path / "src" / "b.txt"
    a_file.parent.mkdir(parents=True, exist_ok=True)
    a_file.write_text("a0\nhold\n", encoding="utf-8")
    b_file.write_text("b0\nhold\n", encoding="utf-8")

    diff = (
        "diff --git a/src/a.txt b/src/a.txt\n"
        "--- a/src/a.txt\n"
        "+++ b/src/a.txt\n"
        "@@ -1,2 +1,2 @@\n"
        "-a0\n"
        "+a1\n"
        " hold\n"
        "diff --git a/src/b.txt b/src/b.txt\n"
        "--- a/src/b.txt\n"
        "+++ b/src/b.txt\n"
        "@@ -1,2 +1,2 @@\n"
        "-b0\n"
        "+b1\n"
        " hold\n"
    )

    _set_which(monkeypatch, git=True, perl=False)
    _force_git_failure(monkeypatch)

    real_safe_write = apply_patch_mod.safe_write_file

    def _failing_safe_write(*, repo_root: Path, path: str, content: str, allowed_prefixes):
        if path == "src/b.txt":
            return {"ok": False, "path": path, "error": "forced write failure"}
        return real_safe_write(repo_root=repo_root, path=path, content=content, allowed_prefixes=allowed_prefixes)

    monkeypatch.setattr(apply_patch_mod, "safe_write_file", _failing_safe_write)

    result = safe_apply_patch(repo_root=tmp_path, diff=diff, check_only=False)

    assert result["ok"] is False
    assert "rollback" in result["error"].lower()
    assert a_file.read_text(encoding="utf-8") == "a0\nhold\n"
    assert b_file.read_text(encoding="utf-8") == "b0\nhold\n"


def test_perl_fallback_failure_rolls_back_partial_mutation(tmp_path: Path, monkeypatch) -> None:
    a_file = tmp_path / "src" / "a.txt"
    b_file = tmp_path / "src" / "b.txt"
    a_file.parent.mkdir(parents=True, exist_ok=True)
    a_file.write_text("a0\n", encoding="utf-8")
    b_file.write_text("b0\n", encoding="utf-8")

    diff = (
        "diff --git a/src/a.txt b/src/a.txt\n"
        "--- a/src/a.txt\n"
        "+++ b/src/a.txt\n"
        "@@ -1 +1 @@\n"
        "-a0\n"
        "+a1\n"
        "diff --git a/src/b.txt b/src/b.txt\n"
        "--- a/src/b.txt\n"
        "+++ b/src/b.txt\n"
        "@@ -1 +1 @@\n"
        "-b0\n"
        "+b1\n"
    )

    _set_which(monkeypatch, git=True, perl=True)
    _force_git_failure(monkeypatch)

    def _partial_perl(*, repo_root: Path, parsed_files, check_only: bool):
        _ = parsed_files
        if not check_only:
            (repo_root / "src" / "a.txt").write_text("a1\n", encoding="utf-8")
        return False, "forced perl failure", "", ""

    monkeypatch.setattr(apply_patch_mod, "_run_perl_fallback", _partial_perl)
    monkeypatch.setattr(apply_patch_mod, "_compute_python_fallback", lambda **_kwargs: (False, {}, "python blocked"))

    result = safe_apply_patch(repo_root=tmp_path, diff=diff, check_only=False)

    assert result["ok"] is False
    perl_attempts = [x for x in result["attempts"] if x["strategy"] == "perl"]
    assert perl_attempts
    assert "rollback" in perl_attempts[0]["detail"].lower()
    assert a_file.read_text(encoding="utf-8") == "a0\n"
    assert b_file.read_text(encoding="utf-8") == "b0\n"


def test_rejects_unsupported_rename_copy_mode_binary_features(tmp_path: Path) -> None:
    diff = (
        "diff --git a/src/old.txt b/src/new.txt\n"
        "similarity index 100%\n"
        "rename from src/old.txt\n"
        "rename to src/new.txt\n"
    )

    result = safe_apply_patch(repo_root=tmp_path, diff=diff, check_only=True)

    assert result["ok"] is False
    assert "unsupported diff feature" in result["error"].lower()


def test_no_delete_enforcement_blocks_dev_null_deletion(tmp_path: Path) -> None:
    diff = (
        "diff --git a/src/dead.txt b/src/dead.txt\n"
        "--- a/src/dead.txt\n"
        "+++ /dev/null\n"
        "@@ -1 +0,0 @@\n"
        "-gone\n"
    )

    result = safe_apply_patch(repo_root=tmp_path, diff=diff, check_only=False)

    assert result["ok"] is False
    assert "deletes files" in result["error"].lower()


def test_path_safety_blocks_traversal_and_disallowed_prefix(tmp_path: Path) -> None:
    traversal_diff = (
        "diff --git a/../evil.txt b/../evil.txt\n"
        "--- a/../evil.txt\n"
        "+++ b/../evil.txt\n"
        "@@ -1 +1 @@\n"
        "-x\n"
        "+y\n"
    )

    traversal_result = safe_apply_patch(repo_root=tmp_path, diff=traversal_diff, check_only=True)
    assert traversal_result["ok"] is False
    assert "traversal" in traversal_result["error"].lower()

    prefix_diff = (
        "diff --git a/docs/readme.txt b/docs/readme.txt\n"
        "--- a/docs/readme.txt\n"
        "+++ b/docs/readme.txt\n"
        "@@ -1 +1 @@\n"
        "-a\n"
        "+b\n"
    )
    prefix_result = safe_apply_patch(
        repo_root=tmp_path,
        diff=prefix_diff,
        allowed_prefixes=("src/",),
        check_only=True,
    )
    assert prefix_result["ok"] is False
    assert "disallowed path" in prefix_result["error"].lower()


def test_attempt_order_and_single_attempt_per_non_git_strategy(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "src" / "demo.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("old\nkeep\n", encoding="utf-8")

    diff = (
        "diff --git a/src/demo.txt b/src/demo.txt\n"
        "--- a/src/demo.txt\n"
        "+++ b/src/demo.txt\n"
        "@@ -1,2 +1,2 @@\n"
        "-old\n"
        "+new\n"
        " keep\n"
    )

    _set_which(monkeypatch, git=True, perl=False)
    _force_git_failure(monkeypatch)

    result = safe_apply_patch(repo_root=tmp_path, diff=diff, check_only=False)

    assert result["ok"] is True
    attempts = result["attempts"]
    strategies = [item["strategy"] for item in attempts]

    assert strategies[0] == "git"
    assert strategies.count("perl") == 1
    assert strategies.count("python") == 1
    assert strategies.count("write_file") == 1
    assert strategies.index("perl") < strategies.index("python") < strategies.index("write_file")


def test_strategy_hint_python_skips_git_and_perl(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "src" / "demo.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("old\nkeep\n", encoding="utf-8")

    diff = (
        "diff --git a/src/demo.txt b/src/demo.txt\n"
        "--- a/src/demo.txt\n"
        "+++ b/src/demo.txt\n"
        "@@ -1,2 +1,2 @@\n"
        "-old\n"
        "+new\n"
        " keep\n"
    )

    _set_which(monkeypatch, git=True, perl=True)
    result = safe_apply_patch(repo_root=tmp_path, diff=diff, strategy_hint="python", strict_strategy=False)

    assert result["ok"] is True
    assert result["strategy_requested"] == "python"
    assert result["strategy"] == "write_file"
    strategies = [item["strategy"] for item in result["attempts"]]
    assert "git" not in strategies
    assert "perl" not in strategies
    assert "python" in strategies
    assert "write_file" in strategies


def test_strict_strategy_python_does_not_fall_forward(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "src" / "demo.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("old\nkeep\n", encoding="utf-8")

    diff = (
        "diff --git a/src/demo.txt b/src/demo.txt\n"
        "--- a/src/demo.txt\n"
        "+++ b/src/demo.txt\n"
        "@@ -1,2 +1,2 @@\n"
        "-old\n"
        "+new\n"
        " keep\n"
    )

    _set_which(monkeypatch, git=True, perl=True)
    monkeypatch.setattr(
        apply_patch_mod,
        "_compute_python_fallback",
        lambda **_kwargs: (False, {}, "forced python failure"),
    )

    result = safe_apply_patch(repo_root=tmp_path, diff=diff, strategy_hint="python", strict_strategy=True)
    assert result["ok"] is False
    assert result["strategy_requested"] == "python"
    assert result["strategy"] == "python"
    strategies = [item["strategy"] for item in result["attempts"]]
    assert strategies == ["python"]


def test_fallback_preserves_no_trailing_newline_marker(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "src" / "demo.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("old", encoding="utf-8")

    diff = (
        "diff --git a/src/demo.txt b/src/demo.txt\n"
        "--- a/src/demo.txt\n"
        "+++ b/src/demo.txt\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "\\ No newline at end of file\n"
        "+new\n"
        "\\ No newline at end of file\n"
    )

    _set_which(monkeypatch, git=True, perl=False)
    _force_git_failure(monkeypatch)

    result = safe_apply_patch(repo_root=tmp_path, diff=diff, check_only=False)
    assert result["ok"] is True
    assert result["strategy"] == "write_file"
    assert target.read_text(encoding="utf-8") == "new"
