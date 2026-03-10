"""
mana_analyzer.tools.apply_patch

A safe patch-application tool for coding agents.

Key properties:
- Parses the patch to identify touched paths.
- Refuses patches that touch files outside repo_root.
- Optionally restricts touched paths to allowed prefixes.
- Applies patch using a deterministic strategy chain:
  shell (git apply / patch) -> python compute -> write_file persistence.
- Intended usage: run patch validation (`check_only=true`) first, then apply.
- NO-DELETE: explicitly blocks patches that delete files (/dev/null targets).
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Optional, Sequence

from .write_file import safe_write_file

# None => allow any path under repo_root
DEFAULT_ALLOWED_PREFIXES: Optional[tuple[str, ...]] = None

_DRIVE_LETTER_RE = re.compile(r"^[a-zA-Z]:[\\/]")
_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


# Guidance to reduce common patch-shape failures.
PATCH_FORMAT_GUIDANCE = (
    "Patch must be a VALID unified diff payload, e.g.:\n"
    "  diff --git a/path/to/file.py b/path/to/file.py\n"
    "  --- a/path/to/file.py\n"
    "  +++ b/path/to/file.py\n"
    "  @@ -1,3 +1,4 @@\n"
    "Do NOT use '*** Begin Patch'/'*** End Patch' format. "
    "Do NOT use absolute paths or '..'."
)


@dataclass(frozen=True)
class ApplyPatchResult:
    ok: bool
    touched_files: list[str]
    strip_level: int = -1
    check_only: bool = False
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    strategy_requested: str = "auto"
    strategy: str = ""
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _DiffLine:
    op: str
    text: str


@dataclass(frozen=True)
class _DiffHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[_DiffLine]


@dataclass(frozen=True)
class _DiffFile:
    old_path: str
    new_path: str
    hunks: list[_DiffHunk]
    new_has_trailing_newline: bool = True


@dataclass(frozen=True)
class _FileSnapshot:
    existed: bool
    content: str


def _strip_markdown_fences(text: str) -> str:
    """
    If an LLM wraps diffs in
```diff ... 
``` or
``` ... 
```, strip the fences.
    This keeps the tool tolerant without changing the semantics of the patch.
    """
    s = text.strip()
    if not s.startswith("```"):
        return text
    lines = s.splitlines()
    if len(lines) < 2:
        return text
    if lines[-1].strip() != "```":
        return text
    # drop first and last fence line
    inner = "\n".join(lines[1:-1])
    return inner.strip() + "\n"


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


def _is_dev_null(path: str) -> bool:
    return path in {"dev/null", "/dev/null"}


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


def _parse_unified_diff_strict(diff_text: str) -> tuple[list[_DiffFile], str]:
    lines = diff_text.splitlines()
    parsed: list[_DiffFile] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if not line.startswith("diff --git "):
            # Ignore prelude lines generated by LLM conversations
            i += 1
            continue

        parts = line.split()
        if len(parts) < 4:
            return [], "Error: malformed diff --git header"

        block_old_path = _normalise_patch_path(parts[2])
        block_new_path = _normalise_patch_path(parts[3])
        i += 1

        header_old = block_old_path
        header_new = block_new_path
        saw_old = False
        saw_new = False
        saw_mode_metadata = False
        new_has_trailing_newline = True
        hunks: list[_DiffHunk] = []

        while i < len(lines) and not lines[i].startswith("diff --git "):
            current = lines[i]

            if current.startswith(("rename from ", "rename to ", "copy from ", "copy to ", "similarity index ")):
                return [], "Error: unsupported diff feature (rename/copy)"
            if current.startswith(("GIT binary patch", "Binary files ")):
                return [], "Error: unsupported diff feature (binary patch)"
            if current.startswith(("old mode ", "new mode ", "deleted file mode ", "new file mode ")):
                saw_mode_metadata = True
                i += 1
                continue
            if current.startswith("index "):
                i += 1
                continue
            if current.startswith("--- "):
                header_old = _normalise_patch_path(current[4:])
                saw_old = True
                i += 1
                continue
            if current.startswith("+++ "):
                header_new = _normalise_patch_path(current[4:])
                saw_new = True
                i += 1
                continue
            if current.startswith("@@ "):
                if not saw_old or not saw_new:
                    return [], "Error: malformed diff block (missing ---/+++ before hunk)"
                m = _HUNK_HEADER_RE.match(current)
                if m is None:
                    return [], f"Error: malformed hunk header: {current[:120]}"

                old_start = int(m.group(1))
                new_start = int(m.group(3))
                i += 1

                hunk_lines: list[_DiffLine] = []
                last_hunk_op: str | None = None
                while i < len(lines):
                    hline = lines[i]
                    if hline.startswith("diff --git ") or hline.startswith("@@ "):
                        break
                    if hline.startswith("\\ No newline at end of file"):
                        if last_hunk_op == "+":
                            new_has_trailing_newline = False
                        i += 1
                        continue
                    if not hline:
                        # Empty line, assume context line missing a space character.
                        hunk_lines.append(_DiffLine(op=" ", text=""))
                        last_hunk_op = " "
                        i += 1
                        continue
                    op = hline[0]
                    if op not in {" ", "+", "-"}:
                        # Tolerate context line missing leading space
                        op = " "
                        hunk_lines.append(_DiffLine(op=op, text=hline))
                        last_hunk_op = op
                        i += 1
                        continue
                    hunk_lines.append(_DiffLine(op=op, text=hline[1:]))
                    last_hunk_op = op
                    i += 1

                # Dynamically count to tolerate LLM hallucinated header line counts
                old_seen = sum(1 for x in hunk_lines if x.op in {" ", "-"})
                new_seen = sum(1 for x in hunk_lines if x.op in {" ", "+"})

                hunks.append(
                    _DiffHunk(
                        old_start=old_start,
                        old_count=old_seen,
                        new_start=new_start,
                        new_count=new_seen,
                        lines=hunk_lines,
                    )
                )
                continue
            
            # Skip unrecognized conversation lines between hunks
            i += 1

        if not saw_old or not saw_new:
            if saw_mode_metadata:
                return [], "Error: unsupported diff feature (mode-only diff)"
            return [], "Error: malformed diff block (missing ---/+++ headers)"
        if not hunks:
            return [], "Error: unsupported diff block without hunks"

        parsed.append(
            _DiffFile(
                old_path=header_old,
                new_path=header_new,
                hunks=hunks,
                new_has_trailing_newline=new_has_trailing_newline,
            )
        )

    if not parsed:
        return [], "Error: no valid file diff blocks found"

    return parsed, ""


def _attempt(strategy: str, phase: str, ok: bool, detail: str) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "phase": phase,
        "ok": bool(ok),
        "detail": str(detail or ""),
    }


def _target_rel_path(file_diff: _DiffFile) -> str:
    if not _is_dev_null(file_diff.new_path):
        return _normalise_user_path(file_diff.new_path)
    return _normalise_user_path(file_diff.old_path)


def _lines_to_text(lines: Sequence[str], *, trailing_newline: bool) -> str:
    if not lines:
        return ""
    text = "\n".join(lines)
    if trailing_newline:
        text += "\n"
    return text


def _apply_hunks_to_lines(*, base_lines: Sequence[str], hunks: Sequence[_DiffHunk], file_path: str) -> tuple[bool, list[str], str]:
    out = list(base_lines)
    delta = 0

    for idx, hunk in enumerate(hunks, start=1):
        pos = hunk.old_start - 1 + delta
        if pos < 0 or pos > len(out):
            return False, out, f"hunk {idx}: expected position out of range for {file_path}"

        cursor = pos
        for line in hunk.lines:
            if line.op in {" ", "-"}:
                if cursor >= len(out):
                    return False, out, f"hunk {idx}: context mismatch at EOF for {file_path}"
                if out[cursor] != line.text:
                    return False, out, f"hunk {idx}: context mismatch at line {cursor + 1} for {file_path}. Expected {line.text!r}, got {out[cursor]!r}"
                cursor += 1

        replacement = [line.text for line in hunk.lines if line.op in {" ", "+"}]
        out[pos : pos + hunk.old_count] = replacement
        delta += len(replacement) - hunk.old_count

    return True, out, ""


def _apply_via_shell(repo_root: Path, diff: str, check_only: bool) -> tuple[bool, str, str]:
    """
    Shims OS-level tools (git apply / patch) to apply patches dynamically.
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".patch", encoding="utf-8") as tmp:
        tmp.write(diff)
        tmp_path = tmp.name

    try:
        # Attempt 1: git apply
        cmd_git = ["git", "apply", "--ignore-space-change", "--ignore-whitespace"]
        if check_only:
            cmd_git.append("--check")
        cmd_git.append(tmp_path)
        
        res_git = subprocess.run(cmd_git, cwd=repo_root, capture_output=True, text=True)
        if res_git.returncode == 0:
            return True, "git apply succeeded\n" + res_git.stdout, res_git.stderr
            
        # Attempt 2: patch
        cmd_patch = ["patch", "-p1", "--no-backup-if-mismatch"]
        if check_only:
            cmd_patch.append("--dry-run")
        cmd_patch.extend(["-i", tmp_path])
        
        res_patch = subprocess.run(cmd_patch, cwd=repo_root, capture_output=True, text=True)
        if res_patch.returncode == 0:
            return True, "patch -p1 succeeded\n" + res_patch.stdout, res_patch.stderr
            
        err_msg = (
            f"Shell tool strategies failed.\n"
            f"git apply error: {res_git.stderr.strip() or res_git.stdout.strip()}\n"
            f"patch error: {res_patch.stderr.strip() or res_patch.stdout.strip()}"
        )
        return False, "", err_msg
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _compute_python_fallback(
    *,
    repo_root: Path,
    parsed_files: Sequence[_DiffFile],
) -> tuple[bool, dict[str, str], str]:
    computed: dict[str, str] = {}
    newline_prefs: dict[str, bool] = {}

    for file_diff in parsed_files:
        rel_target = _target_rel_path(file_diff)
        abs_target = (repo_root / Path(rel_target)).resolve()

        if rel_target in computed:
            base_lines = computed[rel_target].splitlines()
        elif _is_dev_null(file_diff.old_path):
            base_lines: list[str] = []
        else:
            if not abs_target.exists():
                return False, {}, f"python fallback: missing target file {rel_target}"
            base_lines = abs_target.read_text(encoding="utf-8").splitlines()

        ok, patched_lines, err = _apply_hunks_to_lines(base_lines=base_lines, hunks=file_diff.hunks, file_path=rel_target)
        if not ok:
            return False, {}, f"python fallback failed: {err}"

        newline_prefs[rel_target] = bool(file_diff.new_has_trailing_newline)
        computed[rel_target] = _lines_to_text(
            patched_lines,
            trailing_newline=newline_prefs[rel_target],
        )

    return True, computed, "python fallback computed updates"


def _snapshot_files(repo_root: Path, paths: Sequence[str]) -> dict[str, _FileSnapshot]:
    snapshots: dict[str, _FileSnapshot] = {}
    for rel in paths:
        abs_path = (repo_root / Path(rel)).resolve()
        if abs_path.exists():
            snapshots[rel] = _FileSnapshot(existed=True, content=abs_path.read_text(encoding="utf-8"))
        else:
            snapshots[rel] = _FileSnapshot(existed=False, content="")
    return snapshots


def _rollback_files(repo_root: Path, snapshots: dict[str, _FileSnapshot]) -> str:
    errors: list[str] = []
    for rel, snap in snapshots.items():
        abs_path = (repo_root / Path(rel)).resolve()
        try:
            if snap.existed:
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                abs_path.write_text(snap.content, encoding="utf-8")
            else:
                if abs_path.exists():
                    abs_path.unlink()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{rel}: {exc}")
    return "; ".join(errors)


def _apply_via_write_file(
    *,
    repo_root: Path,
    computed: dict[str, str],
    allowed_prefixes: Optional[Sequence[str]],
    order: Sequence[str],
) -> tuple[bool, str, str, str]:
    snapshots = _snapshot_files(repo_root, order)

    for rel in order:
        result = safe_write_file(
            repo_root=repo_root,
            path=rel,
            content=computed[rel],
            allowed_prefixes=allowed_prefixes,
        )
        if not bool(result.get("ok")):
            rollback_err = _rollback_files(repo_root, snapshots)
            base_err = str(result.get("error", "write_file fallback failed")).strip() or "write_file fallback failed"
            if rollback_err:
                return False, f"{base_err}; rollback failed: {rollback_err}", "", ""
            return False, f"{base_err}; rollback applied", "", ""

    return True, "write_file fallback applied", "", ""


def safe_apply_patch(
    *,
    repo_root: Path,
    diff: str,
    allowed_prefixes: Optional[Sequence[str]] = DEFAULT_ALLOWED_PREFIXES,
    check_only: bool = False,
    strategy_hint: str = "auto",
    strict_strategy: bool = False,
) -> dict[str, Any]:
    requested_strategy = str(strategy_hint or "auto").strip().lower() or "auto"
    strict_strategy = bool(strict_strategy)
    supported_strategies = ("auto", "shell", "python", "write_file")
    
    if requested_strategy not in supported_strategies:
        return ApplyPatchResult(
            ok=False,
            touched_files=[],
            strategy_requested=requested_strategy,
            error=f"Error: invalid strategy_hint '{requested_strategy}'. Expected one of {supported_strategies}.",
        ).to_dict()

    if not diff.strip():
        return ApplyPatchResult(
            ok=False,
            touched_files=[],
            strategy_requested=requested_strategy,
            error="Error: empty diff",
        ).to_dict()

    # Common LLM behavior: wrap in markdown fences
    diff = _strip_markdown_fences(diff)

    # Reject non-unified patch wrappers early with a clear error.
    head = diff.lstrip()[:200]
    if head.startswith("*** Begin Patch") or "*** Begin Patch" in head:
        return ApplyPatchResult(
            ok=False,
            touched_files=[],
            strategy_requested=requested_strategy,
            error="Error: patch wrapper format is not supported. " + PATCH_FORMAT_GUIDANCE,
        ).to_dict()

    # Another early signal: if it doesn't mention diff --git anywhere, it's very likely invalid.
    if "diff --git " not in diff:
        return ApplyPatchResult(
            ok=False,
            touched_files=[],
            strategy_requested=requested_strategy,
            error="Error: patch missing 'diff --git' header. " + PATCH_FORMAT_GUIDANCE,
        ).to_dict()

    repo_root = repo_root.resolve()

    touched, deleted = _extract_touched_paths_and_deletes(diff)

    # NO-DELETE: refuse any deletion
    if deleted:
        return ApplyPatchResult(
            ok=False,
            touched_files=sorted(touched),
            strategy_requested=requested_strategy,
            error=f"Blocked: patch deletes files (not allowed): {sorted(deleted)}",
        ).to_dict()

    ok, touched_files, err = _validate_touched_paths(repo_root, touched, allowed_prefixes)
    if not ok:
        return ApplyPatchResult(
            ok=False,
            touched_files=[],
            strategy_requested=requested_strategy,
            error=err,
        ).to_dict()

    attempts: list[dict[str, Any]] = []

    # Supported strategy ordering for explicit hints includes shell.
    # The default "auto" order is python -> write_file (deterministic, no subprocess).
    # Shell is only used when explicitly requested via strategy_hint.
    strategy_order = ["shell", "python", "write_file"]
    auto_order = ["python", "write_file"]

    if requested_strategy == "auto":
        run_order = list(auto_order)
    else:
        start = strategy_order.index(requested_strategy)
        run_order = [requested_strategy] if strict_strategy else strategy_order[start:]

    if check_only and requested_strategy == "write_file" and strict_strategy:
        return ApplyPatchResult(
            ok=False,
            touched_files=touched_files,
            check_only=True,
            strategy_requested=requested_strategy,
            strategy="write_file",
            attempts=attempts,
            error="Error: strict write_file strategy cannot run in check_only mode.",
        ).to_dict()

    if "write_file" in run_order and "python" not in run_order:
        write_idx = run_order.index("write_file")
        run_order.insert(write_idx, "python")

    # 1. Shell OS Tools approach
    if "shell" in run_order:
        phase = "check" if check_only else "apply"
        shell_ok, shell_stdout, shell_stderr = _apply_via_shell(repo_root, diff, check_only)
        attempts.append(_attempt("shell", phase, shell_ok, shell_stderr if not shell_ok else shell_stdout))
        
        if shell_ok:
            return ApplyPatchResult(
                ok=True,
                touched_files=touched_files,
                check_only=check_only,
                strategy_requested=requested_strategy,
                strategy="shell",
                attempts=attempts,
                stdout=shell_stdout,
                stderr=shell_stderr,
            ).to_dict()
            
        elif strict_strategy and requested_strategy == "shell":
            return ApplyPatchResult(
                ok=False,
                touched_files=touched_files,
                check_only=check_only,
                strategy_requested=requested_strategy,
                strategy="shell",
                attempts=attempts,
                error=f"Error: strict shell strategy failed.\n{shell_stderr}",
            ).to_dict()

    # If shell fails, or was skipped, we proceed to parse the diff for python fallback.
    parsed_files, parse_err = _parse_unified_diff_strict(diff)

    computed: dict[str, str] = {}
    python_available = False

    # 2. Deterministic Python compute
    if "python" in run_order:
        python_phase = "check" if check_only else "compute"
        if parse_err:
            py_ok = False
            py_detail = f"Parser error: {parse_err}"
            attempts.append(_attempt("python", python_phase, py_ok, py_detail))
            python_available = False
            # Unsupported diff features (rename/copy/binary/mode-only) cannot be
            # handled by any fallback strategy; return immediately with the parse error.
            if "unsupported diff feature" in parse_err.lower():
                return ApplyPatchResult(
                    ok=False,
                    touched_files=touched_files,
                    check_only=check_only,
                    strategy_requested=requested_strategy,
                    strategy="python",
                    attempts=attempts,
                    error=parse_err,
                ).to_dict()
            if strict_strategy and requested_strategy == "python":
                return ApplyPatchResult(
                    ok=False,
                    touched_files=touched_files,
                    check_only=check_only,
                    strategy_requested=requested_strategy,
                    strategy="python",
                    attempts=attempts,
                    error=f"Error: patch fallback chain exhausted. {py_detail}",
                ).to_dict()
            if "write_file" not in run_order:
                return ApplyPatchResult(
                    ok=False,
                    touched_files=touched_files,
                    check_only=check_only,
                    strategy_requested=requested_strategy,
                    strategy="python",
                    attempts=attempts,
                    error=f"Error: patch fallback chain exhausted. {py_detail}",
                ).to_dict()
        else:
            py_ok, computed, py_detail = _compute_python_fallback(repo_root=repo_root, parsed_files=parsed_files)
            attempts.append(_attempt("python", python_phase, py_ok, py_detail))
            python_available = py_ok
            if not py_ok:
                if strict_strategy and requested_strategy == "python":
                    return ApplyPatchResult(
                        ok=False,
                        touched_files=touched_files,
                        check_only=check_only,
                        strategy_requested=requested_strategy,
                        strategy="python",
                        attempts=attempts,
                        error=f"Error: patch fallback chain exhausted. {py_detail}",
                    ).to_dict()
                if "write_file" not in run_order:
                    return ApplyPatchResult(
                        ok=False,
                        touched_files=touched_files,
                        check_only=check_only,
                        strategy_requested=requested_strategy,
                        strategy="python",
                        attempts=attempts,
                        error=f"Error: patch fallback chain exhausted. {py_detail}",
                    ).to_dict()
            elif check_only:
                return ApplyPatchResult(
                    ok=True,
                    touched_files=touched_files,
                    check_only=True,
                    strategy_requested=requested_strategy,
                    strategy="python",
                    attempts=attempts,
                    stdout="python fallback check succeeded",
                ).to_dict()

    # 3. Persistence via Write-file fallback
    if "write_file" in run_order:
        if check_only:
            return ApplyPatchResult(
                ok=python_available,
                touched_files=touched_files,
                check_only=True,
                strategy_requested=requested_strategy,
                strategy="python" if python_available else "write_file",
                attempts=attempts,
                error="" if python_available else "Error: write_file strategy requires python compute step.",
                stdout="python fallback check succeeded" if python_available else "",
            ).to_dict()

        if not python_available:
            return ApplyPatchResult(
                ok=False,
                touched_files=touched_files,
                check_only=False,
                strategy_requested=requested_strategy,
                strategy="python",
                attempts=attempts,
                error="Error: python compute prerequisite failed.",
            ).to_dict()

        write_order: list[str] = []
        seen_order: set[str] = set()
        for item in parsed_files:
            rel = _target_rel_path(item)
            if rel in seen_order:
                continue
            seen_order.add(rel)
            write_order.append(rel)
            
        write_ok, write_detail, write_stdout, write_stderr = _apply_via_write_file(
            repo_root=repo_root,
            computed=computed,
            allowed_prefixes=allowed_prefixes,
            order=write_order,
        )
        attempts.append(_attempt("write_file", "apply", write_ok, write_detail))

        if write_ok:
            return ApplyPatchResult(
                ok=True,
                touched_files=touched_files,
                check_only=False,
                strategy_requested=requested_strategy,
                strategy="write_file",
                attempts=attempts,
                stdout=write_stdout,
                stderr=write_stderr,
            ).to_dict()

        return ApplyPatchResult(
            ok=False,
            touched_files=touched_files,
            check_only=False,
            strategy_requested=requested_strategy,
            strategy="write_file",
            attempts=attempts,
            error=f"Error: {write_detail}",
        ).to_dict()

    return ApplyPatchResult(
        ok=False,
        touched_files=touched_files,
        check_only=check_only,
        strategy_requested=requested_strategy,
        strategy=requested_strategy if requested_strategy != "auto" else "",
        attempts=attempts,
        error="Error: No runnable strategy selected.",
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
        strategy_hint: str = "auto",
        strict_strategy: bool = False,
    ) -> dict[str, Any]:
        # Compatibility: some tool-calling models send `patch` instead of `diff`.
        effective_diff = diff if diff is not None else patch
        if effective_diff is None:
            return ApplyPatchResult(
                ok=False,
                touched_files=[],
                check_only=check_only,
                strategy_requested=str(strategy_hint or "auto"),
                error="Error: missing patch content (expected `diff` or `patch`). " + PATCH_FORMAT_GUIDANCE,
            ).to_dict()
        return safe_apply_patch(
            repo_root=repo_root,
            diff=effective_diff,
            allowed_prefixes=allowed_prefixes,
            check_only=check_only,
            strategy_hint=str(strategy_hint or "auto"),
            strict_strategy=bool(strict_strategy),
        )

    return StructuredTool.from_function(
        func=_tool,
        name="apply_patch",
        description=(
            "Safely apply a unified diff patch inside the repository. "
            "Refuses absolute paths, traversal, and paths escaping the repository root. "
            "NO-DELETE: blocks patches that delete files. "
            "Internal strategy chain: shell OS tools -> python compute -> write_file persistence. "
            "Optional strategy controls: strategy_hint=auto|shell|python|write_file and strict_strategy=true|false. "
            "With check_only=true, validation runs through tools without writing files. "
            + PATCH_FORMAT_GUIDANCE
        ),
    )
