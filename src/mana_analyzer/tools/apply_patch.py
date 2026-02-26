"""
mana_analyzer.tools.apply_patch

A safe patch-application tool for coding agents.

Key properties:
- Parses the patch to identify touched paths.
- Refuses patches that touch files outside repo_root.
- Optionally restricts touched paths to allowed prefixes.
- Applies patch using `git apply` (preferred) without `--unsafe-paths`.
- Intended usage: run patch validation (`git apply --check`) first, then apply.
- NO-DELETE: explicitly blocks patches that delete files (/dev/null targets).
- If repeated patch attempts fail or produce no file changes, callers should switch to
  non-patch fallback editing (e.g. write_file) and stop retry loops.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

# None => allow any path under repo_root
DEFAULT_ALLOWED_PREFIXES: Optional[tuple[str, ...]] = None

_DRIVE_LETTER_RE = re.compile(r"^[a-zA-Z]:[\\/]")


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


def _normalise_user_path(path: str) -> str:
    p = path.replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    while p.startswith("/"):
        p = p[1:]
    while "//" in p:
        p = p.replace("//", "/")
    return p


def _normalise_prefixes(prefixes: Optional[Sequence[str]]) -> Optional[tuple[str, ...]]:
    if not prefixes:
        return None
    out: list[str] = []
    for raw in prefixes:
        p = _normalise_user_path(raw)
        if p and not p.endswith("/"):
            p += "/"
        out.append(p)
    return tuple(out)


def _is_allowed_prefix(rel_posix: str, allowed_prefixes: Optional[Sequence[str]]) -> bool:
    if not allowed_prefixes:
        return True
    rel_posix = _normalise_user_path(rel_posix)
    norm = _normalise_prefixes(allowed_prefixes)
    if not norm:
        return True
    for prefix in norm:
        if prefix == "":
            return True
        if rel_posix == prefix[:-1] or rel_posix.startswith(prefix):
            return True
    return False


def _normalise_patch_path(p: str) -> str:
    p = p.strip()
    p = p.split("\t", 1)[0]  # drop timestamps
    if p.startswith(("a/", "b/")):
        p = p[2:]
    return _normalise_user_path(p)


def _extract_touched_paths_and_deletes(diff_text: str) -> tuple[set[str], set[str]]:
    """
    Returns:
      touched: all non-/dev/null paths mentioned (including created files)
      deleted: paths that appear to be deleted by the patch
    """
    touched: set[str] = set()
    deleted: set[str] = set()

    last_old: Optional[str] = None
    last_new: Optional[str] = None

    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                a_path = _normalise_patch_path(parts[2])
                b_path = _normalise_patch_path(parts[3])
                # reset for this file block
                last_old, last_new = a_path, b_path
                if a_path and a_path != "dev/null":
                    touched.add(a_path)
                if b_path and b_path != "dev/null":
                    touched.add(b_path)

        elif line.startswith("--- "):
            last_old = _normalise_patch_path(line[4:])
            if last_old and last_old != "dev/null":
                touched.add(last_old)

        elif line.startswith("+++ "):
            last_new = _normalise_patch_path(line[4:])
            if last_new and last_new != "dev/null":
                touched.add(last_new)

            # Detect deletion: +++ /dev/null
            if last_new in ("dev/null", "/dev/null"):
                if last_old and last_old not in ("dev/null", "/dev/null"):
                    deleted.add(last_old)

    return touched, deleted


def _validate_touched_paths(
    repo_root: Path,
    touched: set[str],
    allowed_prefixes: Optional[Sequence[str]],
) -> tuple[bool, list[str], str]:
    repo_root = repo_root.resolve()
    validated: list[str] = []

    for p in sorted(touched):
        raw = p.strip()

        if "\x00" in raw:
            return False, [], f"Blocked: NUL byte in patch path: {p}"

        if _DRIVE_LETTER_RE.match(raw):
            return False, [], f"Blocked: drive-letter path in patch: {p}"

        # Absolute posix path
        if raw.startswith("/"):
            return False, [], f"Blocked: absolute path in patch: {p}"

        # Traversal segments
        parts = [seg for seg in raw.replace("\\", "/").split("/") if seg not in ("", ".")]
        if any(seg == ".." for seg in parts):
            return False, [], f"Blocked: traversal ('..') in patch path: {p}"

        # Normalise via PurePosixPath to keep semantics stable cross-platform.
        rel_pp = PurePosixPath(_normalise_user_path(raw))
        if str(rel_pp) in ("", "."):
            return False, [], "Blocked: empty/invalid path in patch"

        target = (repo_root / Path(str(rel_pp))).resolve()
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

    touched, deleted = _extract_touched_paths_and_deletes(diff)

    # NO-DELETE: refuse any deletion
    if deleted:
        return ApplyPatchResult(
            ok=False,
            touched_files=sorted(touched),
            error=f"Blocked: patch deletes files (not allowed): {sorted(deleted)}",
        ).to_dict()

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
            "Refuses absolute paths, traversal, and paths escaping the repository root. "
            "NO-DELETE: blocks patches that delete files. "
            "Run check-only validation first and use non-patch fallback if patch attempts fail repeatedly."
        ),
    )