"""Focused tests for deterministic apply_patch context recovery."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from mana_agent.tools.apply_patch import (
    MAX_PATCH_ATTEMPTS,
    _apply_codex_update_with_recovery,
    safe_apply_patch,
)


def _update_patch(path: str, body: str) -> str:
    body = body.strip("\n")
    return f"*** Begin Patch\n*** Update File: {path}\n{body}\n*** End Patch\n"


def test_stale_line_replaced_after_file_changed(tmp_path: Path) -> None:
    target = tmp_path / "src" / "example.py"
    target.parent.mkdir(parents=True)
    target.write_text("value = 1\nhello world now\nvalue = 2\n", encoding="utf-8")

    result = safe_apply_patch(
        repo_root=tmp_path,
        patch=_update_patch(
            "src/example.py",
            "@@\n-hello world\n+hello universe\n",
        ),
    )

    assert result["ok"] is True
    assert result["strategy"] in {"unique_removed_lines", "reduced_context", "anchored_insert"}
    assert "hello universe" in target.read_text(encoding="utf-8")
    assert "hello world now" not in target.read_text(encoding="utf-8")
    assert result["attempts"]
    assert result["attempts"][0]["strategy"] == "exact_context"
    assert result["attempts"][0]["ok"] is False


def test_markdown_row_inserted_when_table_rows_changed(tmp_path: Path) -> None:
    target = tmp_path / "README.md"
    target.write_text(
        "# Demo\n\n## Documentation\n\n"
        "| Doc | Path |\n"
        "| --- | --- |\n"
        "| Overview | docs/01-overview.md |\n"
        "| NEW | docs/new.md |\n"
        "| Install | docs/02-installation.md |\n"
        "| Analyze | docs/analyze.md |\n",
        encoding="utf-8",
    )
    # Stale context omits the NEW row between Overview and Install.
    result = safe_apply_patch(
        repo_root=tmp_path,
        patch=_update_patch(
            "README.md",
            "@@\n"
            " ## Documentation\n"
            " \n"
            " | Doc | Path |\n"
            " | --- | --- |\n"
            " | Overview | docs/01-overview.md |\n"
            " | Install | docs/02-installation.md |\n"
            "+| Adaptive | docs/adaptive-coding-runtime.md |\n"
            " | Analyze | docs/analyze.md |\n",
        ),
    )

    text = target.read_text(encoding="utf-8")
    assert result["ok"] is True
    assert "| Adaptive | docs/adaptive-coding-runtime.md |" in text
    assert "| Analyze | docs/analyze.md |" in text
    assert "| NEW | docs/new.md |" in text
    assert text.count("| Analyze | docs/analyze.md |") == 1


def test_insertion_uses_unique_heading_and_adjacent_row(tmp_path: Path) -> None:
    target = tmp_path / "README.md"
    target.write_text(
        "# Demo\n\n## Documentation\n\n"
        "| Doc | Path |\n"
        "| --- | --- |\n"
        "| Overview | docs/01-overview.md |\n"
        "| Install | docs/02-installation.md |\n",
        encoding="utf-8",
    )
    result = safe_apply_patch(
        repo_root=tmp_path,
        patch=_update_patch(
            "README.md",
            "@@\n"
            " ## Documentation\n"
            " \n"
            " | Doc | Path |\n"
            " | --- | --- |\n"
            " | Install | docs/02-installation.md |\n"
            "+| Routing | docs/multi-agent-routing.md |\n",
        ),
    )

    text = target.read_text(encoding="utf-8")
    assert result["ok"] is True
    assert result["strategy"] in {"exact_context", "reduced_context", "anchored_insert"}
    assert "| Routing | docs/multi-agent-routing.md |" in text
    assert "| Install | docs/02-installation.md |" in text
    assert text.index("| Install |") < text.index("| Routing |")


def test_already_present_change_is_idempotent_success(tmp_path: Path) -> None:
    target = tmp_path / "README.md"
    target.write_text("# Demo\n\n## Telegram\n", encoding="utf-8")

    result = safe_apply_patch(
        repo_root=tmp_path,
        patch=_update_patch("README.md", "@@\n-## Messaging\n+## Telegram\n"),
    )

    assert result["ok"] is True
    assert result["already_applied"] is True
    assert result["strategy"] == "already_applied"
    assert result.get("files_changed") == []
    assert target.read_text(encoding="utf-8") == "# Demo\n\n## Telegram\n"
    assert target.read_text(encoding="utf-8").count("## Telegram") == 1


def test_whitespace_only_context_drift_is_handled(tmp_path: Path) -> None:
    target = tmp_path / "notes.md"
    target.write_text("# Title\n\n    indented note\n", encoding="utf-8")

    result = safe_apply_patch(
        repo_root=tmp_path,
        patch=_update_patch("notes.md", "@@\n-indented note\n+updated note\n"),
    )

    assert result["ok"] is True
    assert "updated note" in target.read_text(encoding="utf-8")
    assert result["strategy"] in {"unique_removed_lines", "whitespace_normalized", "reduced_context"}


def test_multiple_matching_anchors_produce_ambiguity_error(tmp_path: Path) -> None:
    target = tmp_path / "dup.py"
    target.write_text("token\nkeep\ntoken\n", encoding="utf-8")

    result = safe_apply_patch(
        repo_root=tmp_path,
        patch=_update_patch("dup.py", "@@\n-token\n+changed\n"),
    )

    assert result["ok"] is False
    assert result["error_code"] == "patch_context_not_found"
    assert result["strategy"] == "ambiguous_context"
    assert int(result.get("candidate_count") or 0) >= 2
    assert "ambiguous" in str(result.get("error") or "").lower() or "candidate" in str(result.get("error") or "").lower()
    assert target.read_text(encoding="utf-8") == "token\nkeep\ntoken\n"


def test_original_stale_patch_is_not_submitted_repeatedly(tmp_path: Path) -> None:
    target = tmp_path / "a.py"
    target.write_text("alpha\n", encoding="utf-8")
    stale = _update_patch("a.py", "@@\n-missing line never present\n+beta\n")
    fingerprints: list[str] = []

    original = _apply_codex_update_with_recovery

    def _wrapped(content: str, lines, path: str):  # noqa: ANN001
        from mana_agent.tools.apply_patch import _patch_fingerprint

        fingerprints.append(_patch_fingerprint(lines))
        return original(content, lines, path)

    with patch("mana_agent.tools.apply_patch._apply_codex_update_with_recovery", side_effect=_wrapped):
        result = safe_apply_patch(repo_root=tmp_path, patch=stale)

    assert result["ok"] is False
    # Recovery is invoked once per file; internal attempts reuse rebuilt logic without
    # re-entering safe_apply_patch with the identical stale payload repeatedly.
    assert fingerprints.count(fingerprints[0]) == 1
    attempts = result.get("attempts") or []
    assert len(attempts) <= MAX_PATCH_ATTEMPTS
    assert attempts[0]["ok"] is False
    assert attempts[0]["strategy"] == "exact_context"


def test_retry_count_is_bounded(tmp_path: Path) -> None:
    target = tmp_path / "a.py"
    target.write_text("only this\n", encoding="utf-8")

    result = safe_apply_patch(
        repo_root=tmp_path,
        patch=_update_patch("a.py", "@@\n-totally absent context\n+replacement\n"),
    )

    assert result["ok"] is False
    attempts = result.get("attempts") or []
    assert 1 <= len(attempts) <= MAX_PATCH_ATTEMPTS
    assert all(int(row.get("attempt") or 0) <= MAX_PATCH_ATTEMPTS for row in attempts)
    assert result.get("recovery_error") or result.get("error")


def test_existing_unrelated_rows_are_preserved(tmp_path: Path) -> None:
    target = tmp_path / "README.md"
    target.write_text(
        "## Documentation\n\n"
        "| Doc | Path |\n"
        "| --- | --- |\n"
        "| Overview | docs/01-overview.md |\n"
        "| Analyze | docs/analyze.md |\n"
        "| Browser | docs/17-browser-automation.md |\n",
        encoding="utf-8",
    )
    result = safe_apply_patch(
        repo_root=tmp_path,
        patch=_update_patch(
            "README.md",
            "@@\n"
            " | Overview | docs/01-overview.md |\n"
            "+| Adaptive | docs/adaptive-coding-runtime.md |\n"
            " | Analyze | docs/analyze.md |\n",
        ),
    )
    text = target.read_text(encoding="utf-8")
    assert result["ok"] is True
    assert "| Analyze | docs/analyze.md |" in text
    assert "| Browser | docs/17-browser-automation.md |" in text
    assert "| Adaptive | docs/adaptive-coding-runtime.md |" in text


def test_multiple_hunks_recover_independently(tmp_path: Path) -> None:
    target = tmp_path / "mod.py"
    target.write_text("alpha = 1\nmiddle = 0\nomega = 2\n", encoding="utf-8")
    # First hunk exact; second hunk has drifted removed line.
    result = safe_apply_patch(
        repo_root=tmp_path,
        patch=_update_patch(
            "mod.py",
            "@@\n-alpha = 1\n+alpha = 10\n@@\n-omega = 9\n+omega = 20\n",
        ),
    )
    # omega = 9 is stale (file has omega = 2). Constrained unique similar line should recover.
    text = target.read_text(encoding="utf-8")
    assert result["ok"] is True
    assert "alpha = 10" in text
    assert "middle = 0" in text
    assert "omega = 20" in text


def test_check_only_validates_rebuilt_patch_without_modifying_files(tmp_path: Path) -> None:
    target = tmp_path / "readme.md"
    original = "# Title\n\nhello world now\n"
    target.write_text(original, encoding="utf-8")

    result = safe_apply_patch(
        repo_root=tmp_path,
        patch=_update_patch("readme.md", "@@\n-hello world\n+hello universe\n"),
        check_only=True,
    )

    assert result["ok"] is True
    assert result["check_only"] is True
    assert result.get("files_changed") == []
    assert target.read_text(encoding="utf-8") == original
    assert result["strategy"] in {"unique_removed_lines", "reduced_context", "anchored_insert"}


def test_recovery_metadata_is_populated(tmp_path: Path) -> None:
    target = tmp_path / "meta.md"
    target.write_text("# Heading\n\nstale value here\n", encoding="utf-8")

    result = safe_apply_patch(
        repo_root=tmp_path,
        patch=_update_patch("meta.md", "@@\n-stale value\n+fresh value\n"),
    )

    assert result["ok"] is True
    assert result["strategy"]
    assert isinstance(result.get("attempts"), list) and result["attempts"]
    assert "matched_anchor" in result
    assert "candidate_count" in result
    assert "already_applied" in result
    assert "recovery_error" in result
    assert result.get("changed_ranges")
    assert any(row.get("path") == "meta.md" for row in result["changed_ranges"])


def test_post_apply_verification_detects_missing_expected_result(tmp_path: Path) -> None:
    target = tmp_path / "verify.py"
    target.write_text("keep\n", encoding="utf-8")

    def _fake_recover(content: str, lines, path: str, *, minimal: bool):  # noqa: ANN001
        # Pretend recovery succeeded while producing content that lacks the intended edit.
        return content, {
            "strategy": "unique_removed_lines",
            "matched_anchor": "keep",
            "candidate_count": 1,
            "already_applied": False,
            "error": "",
            "anchors_searched": ["unique_removed_lines:keep"],
            "candidates": [],
            "failed_hunk": "",
            "patch_fingerprint": "fake",
        }

    with patch("mana_agent.tools.apply_patch._recover_update_content", side_effect=_fake_recover):
        result = safe_apply_patch(
            repo_root=tmp_path,
            patch=_update_patch("verify.py", "@@\n-absent\n+expected-line\n"),
        )

    assert result["ok"] is False
    assert "post_apply_verify" in str(result.get("recovery_error") or result.get("error") or "")
    assert target.read_text(encoding="utf-8") == "keep\n"


def test_tool_manager_attaches_reread_on_unrecoverable_context(tmp_path: Path) -> None:
    from mana_agent.multi_agent.core.types import QueueJob, QueueJobType
    from mana_agent.multi_agent.tools.tool_manager import ToolsManager

    (tmp_path / "README.md").write_text("current\n", encoding="utf-8")
    manager = ToolsManager(tmp_path)
    job = QueueJob(
        job_id="j1",
        task_id="t1",
        requested_by_agent_id="a1",
        approved_by_agent_id="a1",
        job_type=QueueJobType.APPLY_PATCH,
        payload={
            "patch": _update_patch("README.md", "@@\n-stale\n+fresh\n"),
        },
        purpose="test",
    )
    # QueueJob may require more fields depending on version; fall back to minimal construct.
    try:
        result = manager.execute_job(job)
    except TypeError:
        job = QueueJob(  # type: ignore[call-arg]
            job_id="j1",
            task_id="t1",
            requested_by_agent_id="a1",
            approved_by_agent_id="a1",
            job_type=QueueJobType.APPLY_PATCH,
            payload={"patch": _update_patch("README.md", "@@\n-stale\n+fresh\n")},
        )
        result = manager.execute_job(job)

    assert result.ok is False
    payload = result.result if isinstance(result.result, dict) else {}
    assert payload.get("error_code") == "patch_context_not_found"
    assert payload.get("reread_files")
    assert payload["reread_files"][0]["path"] == "README.md"
    assert "current" in str(payload["reread_files"][0].get("content") or "")
    assert "reread target file" in str(result.error or "")
