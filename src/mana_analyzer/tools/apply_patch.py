"""
mana_analyzer.tools.apply_patch

A safe patch-application tool for coding agents.

Key properties:
- Parses the patch to identify touched paths.
- Refuses patches that touch files outside repo_root.
- Optionally restricts touched paths to allowed prefixes.
- Applies patch using `git apply` (preferred) without `--unsafe-paths`.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

# None => allow any path under repo_root
DEFAULT_ALLOWED_PREFIXES: Optional[tuple[str, ...]] = None


@dataclass(frozen=True)
class ApplyPatchResult:
    ok: bool
    touched_files: list[str]
    strip_level: int = -1
    check_only: bool = False
    stdout: str = ""
    stderr: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalise_patch_path(p: str) -> str:
    p = p.strip()
    p = p.split("\t", 1)[0]  # drop timestamps
    if p.startswith(("a/", "b/")):
        p = p[2:]
    return p.replace("\\", "/").lstrip("./")


def _extract_touched_paths(diff_text: str) -> set[str]:
    touched: set[str] = set()
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                b_path = _normalise_patch_path(parts[3])
                if b_path != "/dev/null":
                    touched.add(b_path)
        elif line.startswith("+++ ") or line.startswith("--- "):
            p = _normalise_patch_path(line[4:])
            if p and p != "/dev/null":
                touched.add(p)
    return touched


def _is_allowed_prefix(rel_posix: str, allowed_prefixes: Optional[Sequence[str]]) -> bool:
    if not allowed_prefixes:
        return True
    return any(rel_posix.startswith(prefix) for prefix in allowed_prefixes)


def _validate_touched_paths(
    repo_root: Path,
    touched: set[str],
    allowed_prefixes: Optional[Sequence[str]],
) -> tuple[bool, list[str], str]:
    repo_root = repo_root.resolve()
    validated: list[str] = []

    for p in sorted(touched):
        user_path = Path(p)
        if user_path.is_absolute():
            return False, [], f"Blocked: absolute path in patch: {p}"

        target = (repo_root / p).resolve()
        try:
            rel = target.relative_to(repo_root)
        except ValueError:
            return False, [], f"Blocked: patch path escapes repository root: {p}"

        rel_posix = rel.as_posix()
        if not _is_allowed_prefix(rel_posix, allowed_prefixes):
            return False, [], f"Blocked: patch touches disallowed path: {rel_posix}"

        validated.append(rel_posix)

    return True, validated, ""


def _run(cmd: list[str], *, cwd: Path, stdin_text: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        input=stdin_text,
        text=True,
        capture_output=True,
        cwd=str(cwd),
        check=False,
    )


def safe_apply_patch(
    *,
    repo_root: Path,
    diff: str,
    allowed_prefixes: Optional[Sequence[str]] = DEFAULT_ALLOWED_PREFIXES,
    check_only: bool = False,
) -> dict[str, Any]:
    if not diff.strip():
        return ApplyPatchResult(ok=False, touched_files=[], error="Error: empty diff").to_dict()

    repo_root = repo_root.resolve()
    touched = _extract_touched_paths(diff)
    ok, touched_files, err = _validate_touched_paths(repo_root, touched, allowed_prefixes)
    if not ok:
        return ApplyPatchResult(ok=False, touched_files=[], error=err).to_dict()

    if shutil.which("git") is None:
        return ApplyPatchResult(ok=False, touched_files=touched_files, error="Error: git not found on PATH").to_dict()

    last_stderr = ""
    for p in (0, 1, 2):
        check_cmd = ["git", "apply", f"-p{p}", "--check", "-"]
        proc = _run(check_cmd, cwd=repo_root, stdin_text=diff)
        if proc.returncode == 0:
            if check_only:
                return ApplyPatchResult(
                    ok=True,
                    touched_files=touched_files,
                    strip_level=p,
                    check_only=True,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                ).to_dict()

            apply_cmd = ["git", "apply", f"-p{p}", "--whitespace=nowarn", "-"]
            proc2 = _run(apply_cmd, cwd=repo_root, stdin_text=diff)
            if proc2.returncode == 0:
                logger.info("Applied patch with -p%d touching %d files", p, len(touched_files))
                return ApplyPatchResult(
                    ok=True,
                    touched_files=touched_files,
                    strip_level=p,
                    check_only=False,
                    stdout=proc2.stdout,
                    stderr=proc2.stderr,
                ).to_dict()

            return ApplyPatchResult(
                ok=False,
                touched_files=touched_files,
                strip_level=p,
                error="Error: patch check passed but apply failed",
                stdout=proc2.stdout,
                stderr=proc2.stderr,
            ).to_dict()

        last_stderr = proc.stderr or last_stderr

    return ApplyPatchResult(
        ok=False,
        touched_files=touched_files,
        error="Error: patch did not apply cleanly with -p0/-p1/-p2",
        stderr=last_stderr,
    ).to_dict()


def build_apply_patch_tool(*, repo_root: Path, allowed_prefixes: Optional[Sequence[str]] = DEFAULT_ALLOWED_PREFIXES):
    try:
        from langchain_core.tools import StructuredTool  # type: ignore
    except Exception:  # pragma: no cover
        from langchain.tools import StructuredTool  # type: ignore

    def _tool(
        diff: str | None = None,
        patch: str | None = None,
        check_only: bool = False,
    ) -> dict[str, Any]:
        # Compatibility: some tool-calling models send `patch` instead of `diff`.
        effective_diff = diff if diff is not None else patch
        if effective_diff is None:
            return ApplyPatchResult(
                ok=False,
                touched_files=[],
                check_only=check_only,
                error="Error: missing patch content (expected `diff` or `patch`)",
            ).to_dict()
        return safe_apply_patch(
            repo_root=repo_root,
            diff=effective_diff,
            allowed_prefixes=allowed_prefixes,
            check_only=check_only,
        )

    return StructuredTool.from_function(
        func=_tool,
        name="apply_patch",
        description=(
            "Safely apply a unified diff patch inside the repository. "
            "Refuses absolute paths and paths escaping the repository root."
        ),
    )
