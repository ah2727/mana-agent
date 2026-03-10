"""
mana_analyzer.tools.apply_patch

A safe patch-application tool for coding agents.

Key properties:
- Parses the patch to identify touched paths.
- Refuses patches that touch files outside repo_root.
- Optionally restricts touched paths to allowed prefixes.
- Applies patch using a deterministic strategy chain:
  py (Python compute + write_file) -> sh (system patch) -> perl (Perl one-liner) -> auto (best-effort).
- Intended usage: run patch validation (`check_only=true`) first, then apply.
- NO-DELETE: explicitly blocks patches that delete files (/dev/null targets).
- Does NOT use, accept, or depend on git in any way.
"""

from __future__ import annotations

import json
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
    "Patch must be a JSON list of file-edit operations, e.g.:\n"
    '  [{"path": "src/foo.py", "hunks": [{"old_start": 10, "old_lines": ["old line"], "new_lines": ["new line"]}]}]\n'
    "OR a plain unified diff (WITHOUT any git headers), e.g.:\n"
    "  --- a/path/to/file.py\n"
    "  +++ b/path/to/file.py\n"
    "  @@ -1,3 +1,4 @@\n"
    "  context line\n"
    "  -removed line\n"
    "  +added line\n"
    "Do NOT use 'diff --git' headers — they are rejected. "
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
    op: str   # " " for context, "+" for add, "-" for remove
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


# ---------------------------------------------------------------------------
# Patch format detection
# ---------------------------------------------------------------------------

class _PatchFormat:
    """Identifies the format of the incoming patch content."""
    JSON = "json"
    UNIFIED = "unified"   # Standard unified diff (--- / +++ / @@, no git headers)
    UNKNOWN = "unknown"


def _detect_patch_format(text: str) -> str:
    """Detect whether the patch is JSON, plain unified diff, or unknown."""
    stripped = text.strip()

    # Try JSON first
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            json.loads(stripped)
            return _PatchFormat.JSON
        except (json.JSONDecodeError, ValueError):
            pass

    # Check for standard unified diff markers (--- / +++ / @@)
    has_minus = False
    has_plus = False
    has_hunk = False
    for line in stripped.splitlines():
        if line.startswith("--- "):
            has_minus = True
        elif line.startswith("+++ "):
            has_plus = True
        elif line.startswith("@@ "):
            has_hunk = True
        if has_minus and has_plus and has_hunk:
            return _PatchFormat.UNIFIED

    return _PatchFormat.UNKNOWN


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Path extraction and validation
# ---------------------------------------------------------------------------

def _extract_touched_paths_and_deletes_unified(diff_text: str) -> tuple[set[str], set[str]]:
    """
    Extract touched paths from a plain unified diff (no git headers).
    Returns:
      touched: all non-/dev/null paths mentioned (including created files)
      deleted: paths that appear to be deleted by the patch
    """
    touched: set[str] = set()
    deleted: set[str] = set()

    last_old: Optional[str] = None

    for line in diff_text.splitlines():
        if line.startswith("--- "):
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


def _extract_touched_paths_and_deletes_json(patch_data: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    """
    Extract touched paths from a JSON patch format.
    """
    touched: set[str] = set()
    deleted: set[str] = set()

    for entry in patch_data:
        path = _normalise_user_path(str(entry.get("path", "")))
        if path:
            touched.add(path)
        # Check for delete marker
        if entry.get("delete", False):
            deleted.add(path)

    return touched, deleted


def _extract_touched_paths_and_deletes(diff_text: str, fmt: str) -> tuple[set[str], set[str]]:
    """
    Dispatcher: extract touched paths based on detected format.
    """
    if fmt == _PatchFormat.JSON:
        stripped = diff_text.strip()
        data = json.loads(stripped)
        if isinstance(data, dict):
            data = [data]
        return _extract_touched_paths_and_deletes_json(data)
    else:
        return _extract_touched_paths_and_deletes_unified(diff_text)


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


# ---------------------------------------------------------------------------
# Unified diff parser (plain format only, NO git headers)
# ---------------------------------------------------------------------------

def _parse_unified_diff_strict(diff_text: str) -> tuple[list[_DiffFile], str]:
    """
    Parse a plain unified diff (--- / +++ / @@ blocks only).
    Rejects any line starting with 'diff --git'.
    """
    lines = diff_text.splitlines()
    parsed: list[_DiffFile] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Look for --- header to start a new file block
        if not line.startswith("--- "):
            i += 1
            continue

        header_old = _normalise_patch_path(line[4:])
        i += 1

        # Expect +++ immediately after ---
        if i >= len(lines) or not lines[i].startswith("+++ "):
            return [], f"Error: expected '+++ ' after '--- ' at line {i}"

        header_new = _normalise_patch_path(lines[i][4:])
        i += 1

        new_has_trailing_newline = True
        hunks: list[_DiffHunk] = []

        # Parse hunks for this file
        while i < len(lines):
            current = lines[i]

            # If we hit another --- line, it's the next file block
            if current.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
                break

            if current.startswith("@@ "):
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

                    # Stop at next hunk, next file, or end
                    if hline.startswith("@@ "):
                        break
                    if hline.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
                        break

                    if hline.startswith("\\ No newline at end of file"):
                        if last_hunk_op == "+":
                            new_has_trailing_newline = False
                        i += 1
                        continue

                    if not hline:
                        # Empty line, assume context line missing a space character
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

            # Skip unrecognized lines between hunks
            i += 1

        if not hunks:
            return [], "Error: diff block without hunks"

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


# ---------------------------------------------------------------------------
# JSON patch parser
# ---------------------------------------------------------------------------

def _parse_json_patch(text: str) -> tuple[list[_DiffFile], str]:
    """
    Parse JSON patch format into _DiffFile structures.
    Expected format:
    [
      {
        "path": "src/foo.py",
        "create": false,
        "hunks": [
          {
            "old_start": 10,
            "old_lines": ["line to remove or context"],
            "new_lines": ["replacement line"]
          }
        ]
      }
    ]
    """
    try:
        data = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError) as exc:
        return [], f"Error: invalid JSON patch: {exc}"

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        return [], "Error: JSON patch must be a list of file-edit objects"

    parsed: list[_DiffFile] = []

    for entry_idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            return [], f"Error: entry {entry_idx} is not a dict"

        path = _normalise_user_path(str(entry.get("path", "")))
        if not path:
            return [], f"Error: entry {entry_idx} missing 'path'"

        is_create = bool(entry.get("create", False))
        raw_hunks = entry.get("hunks", [])

        if not isinstance(raw_hunks, list) or not raw_hunks:
            return [], f"Error: entry {entry_idx} missing or empty 'hunks'"

        hunks: list[_DiffHunk] = []

        for h_idx, h in enumerate(raw_hunks):
            if not isinstance(h, dict):
                return [], f"Error: entry {entry_idx} hunk {h_idx} is not a dict"

            old_start = int(h.get("old_start", 1))
            old_lines_raw = h.get("old_lines", [])
            new_lines_raw = h.get("new_lines", [])

            if not isinstance(old_lines_raw, list):
                old_lines_raw = [str(old_lines_raw)]
            if not isinstance(new_lines_raw, list):
                new_lines_raw = [str(new_lines_raw)]

            old_lines = [str(line) for line in old_lines_raw]
            new_lines = [str(line) for line in new_lines_raw]

            # Build diff lines: old lines are removals, new lines are additions
            diff_lines: list[_DiffLine] = []
            for ol in old_lines:
                diff_lines.append(_DiffLine(op="-", text=ol))
            for nl in new_lines:
                diff_lines.append(_DiffLine(op="+", text=nl))

            old_count = len(old_lines)
            new_count = len(new_lines)

            hunks.append(
                _DiffHunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=old_start,  # approximate
                    new_count=new_count,
                    lines=diff_lines,
                )
            )

        old_path = "/dev/null" if is_create else path
        new_path = path

        parsed.append(
            _DiffFile(
                old_path=old_path,
                new_path=new_path,
                hunks=hunks,
                new_has_trailing_newline=True,
            )
        )

    if not parsed:
        return [], "Error: no valid entries in JSON patch"

    return parsed, ""


# ---------------------------------------------------------------------------
# Attempt logging helper
# ---------------------------------------------------------------------------

def _attempt(strategy: str, phase: str, ok: bool, detail: str) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "phase": phase,
        "ok": bool(ok),
        "detail": str(detail or ""),
    }


# ---------------------------------------------------------------------------
# Diff application helpers
# ---------------------------------------------------------------------------

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


def _apply_hunks_to_lines(
    *,
    base_lines: Sequence[str],
    hunks: Sequence[_DiffHunk],
    file_path: str,
) -> tuple[bool, list[str], str]:
    out = list(base_lines)
    delta = 0

    for idx, hunk in enumerate(hunks, start=1):
        pos = hunk.old_start - 1 + delta
        if pos < 0 or pos > len(out):
            return False, out, (
                f"hunk {idx}: expected position out of range for {file_path}"
            )

        cursor = pos
        for line in hunk.lines:
            if line.op in {" ", "-"}:
                if cursor >= len(out):
                    return False, out, (
                        f"hunk {idx}: context mismatch at EOF for {file_path}"
                    )
                if out[cursor] != line.text:
                    return False, out, (
                        f"hunk {idx}: context mismatch at line {cursor + 1} for {file_path}. "
                        f"Expected {line.text!r}, got {out[cursor]!r}"
                    )
                cursor += 1

        replacement = [line.text for line in hunk.lines if line.op in {" ", "+"}]
        out[pos : pos + hunk.old_count] = replacement
        delta += len(replacement) - hunk.old_count

    return True, out, ""


# ---------------------------------------------------------------------------
# Strategy: sh (system `patch` command — no git involved)
# ---------------------------------------------------------------------------

def _generate_plain_unified_from_parsed(parsed_files: list[_DiffFile]) -> str:
    """
    Generate a plain unified diff string from parsed _DiffFile structures.
    Used to feed system `patch` utility. Contains only --- / +++ / @@ lines.
    No git headers whatsoever.
    """
    output_lines: list[str] = []

    for file_diff in parsed_files:
        old_path = file_diff.old_path
        new_path = file_diff.new_path

        # Add a/ b/ prefix for patch -p1 compatibility
        if _is_dev_null(old_path):
            output_lines.append("--- /dev/null")
        else:
            output_lines.append(f"--- a/{old_path}")

        if _is_dev_null(new_path):
            output_lines.append("+++ /dev/null")
        else:
            output_lines.append(f"+++ b/{new_path}")

        for hunk in file_diff.hunks:
            output_lines.append(
                f"@@ -{hunk.old_start},{hunk.old_count} +{hunk.new_start},{hunk.new_count} @@"
            )
            for dl in hunk.lines:
                output_lines.append(f"{dl.op}{dl.text}")

        if not file_diff.new_has_trailing_newline:
            output_lines.append("\\ No newline at end of file")

    return "\n".join(output_lines) + "\n"


def _apply_via_sh(
    repo_root: Path,
    parsed_files: list[_DiffFile],
    check_only: bool,
) -> tuple[bool, str, str]:
    """
    Apply patch using system `patch -p1` utility (sh strategy).
    Generates a clean unified diff without any git headers.
    """
    plain_diff = _generate_plain_unified_from_parsed(parsed_files)

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".patch", encoding="utf-8"
    ) as tmp:
        tmp.write(plain_diff)
        tmp_path = tmp.name

    try:
        cmd_patch = ["patch", "-p1", "--no-backup-if-mismatch"]
        if check_only:
            cmd_patch.append("--dry-run")
        cmd_patch.extend(["-i", tmp_path])

        res_patch = subprocess.run(
            cmd_patch, cwd=repo_root, capture_output=True, text=True
        )
        if res_patch.returncode == 0:
            return True, "patch -p1 succeeded\n" + res_patch.stdout, res_patch.stderr

        err_msg = (
            f"sh strategy: patch -p1 failed.\n"
            f"patch error: {res_patch.stderr.strip() or res_patch.stdout.strip()}"
        )
        return False, "", err_msg
    except FileNotFoundError:
        return False, "", "sh strategy: 'patch' command not found on system"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Strategy: py (Python deterministic compute + write_file persistence)
# ---------------------------------------------------------------------------

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
                return False, {}, f"py strategy: missing target file {rel_target}"
            base_lines = abs_target.read_text(encoding="utf-8").splitlines()

        ok, patched_lines, err = _apply_hunks_to_lines(
            base_lines=base_lines, hunks=file_diff.hunks, file_path=rel_target
        )
        if not ok:
            return False, {}, f"py strategy failed: {err}"

        newline_prefs[rel_target] = bool(file_diff.new_has_trailing_newline)
        computed[rel_target] = _lines_to_text(
            patched_lines,
            trailing_newline=newline_prefs[rel_target],
        )

    return True, computed, "py strategy computed updates"


# ---------------------------------------------------------------------------
# Strategy: perl (Perl one-liner fallback for line-by-line edits)
# ---------------------------------------------------------------------------

def _apply_via_perl(
    repo_root: Path,
    parsed_files: list[_DiffFile],
    check_only: bool,
) -> tuple[bool, str, str]:
    """
    Apply patch using Perl one-liner in-place editing.
    Fallback for systems where Python hunk application has issues
    but Perl is available. Uses `perl -i -pe`/`-ne` for in-place substitution.

    For check_only mode, verifies Perl is available and patch is parseable
    without modifying files.
    """
    # Verify perl is available
    try:
        perl_check = subprocess.run(
            ["perl", "-v"], capture_output=True, text=True
        )
        if perl_check.returncode != 0:
            return False, "", "perl strategy: perl not available"
    except FileNotFoundError:
        return False, "", "perl strategy: perl not found on system"

    if check_only:
        return True, "perl strategy: check passed (perl available, patch parseable)", ""

    all_stdout: list[str] = []
    all_stderr: list[str] = []

    for file_diff in parsed_files:
        rel_target = _target_rel_path(file_diff)
        abs_target = (repo_root / Path(rel_target)).resolve()

        if _is_dev_null(file_diff.old_path):
            # New file creation — just write the added lines
            new_content_lines = []
            for hunk in file_diff.hunks:
                for dl in hunk.lines:
                    if dl.op == "+":
                        new_content_lines.append(dl.text)
            content = _lines_to_text(
                new_content_lines,
                trailing_newline=file_diff.new_has_trailing_newline,
            )
            abs_target.parent.mkdir(parents=True, exist_ok=True)
            abs_target.write_text(content, encoding="utf-8")
            all_stdout.append(f"perl strategy: created {rel_target}")
            continue

        if not abs_target.exists():
            return False, "", f"perl strategy: missing target file {rel_target}"

        # For each hunk, build a perl script that does line-range replacement
        # Process hunks in reverse order to avoid line-number shifts
        sorted_hunks = sorted(file_diff.hunks, key=lambda h: h.old_start, reverse=True)

        for h_idx, hunk in enumerate(sorted_hunks):
            new_lines_text = [dl.text for dl in hunk.lines if dl.op in {" ", "+"}]

            start_line = hunk.old_start
            end_line = hunk.old_start + hunk.old_count - 1

            if hunk.old_count == 0:
                # Pure insertion — insert new lines after (start_line - 1)
                insert_text = "\n".join(new_lines_text)
                insert_text_escaped = insert_text.replace("\\", "\\\\").replace("'", "\\'")
                if start_line <= 1:
                    perl_script = (
                        f"BEGIN {{ print '{insert_text_escaped}\\n' }}"
                    )
                else:
                    target_line = start_line - 1
                    perl_script = (
                        f"if ($. == {target_line}) {{ print; "
                        f"print '{insert_text_escaped}\\n'; next }}"
                    )
                cmd = ["perl", "-i", "-pe", perl_script, str(abs_target)]
            else:
                # Replacement: remove old lines, insert new lines
                new_text = "\n".join(new_lines_text)
                new_text_escaped = new_text.replace("\\", "\\\\").replace("'", "\\'")

                if new_lines_text:
                    perl_script = (
                        f"if ($. == {start_line}) {{ print '{new_text_escaped}\\n'; }} "
                        f"elsif ($. > {start_line} && $. <= {end_line}) {{ }} "
                        f"else {{ print; }}"
                    )
                else:
                    # Pure removal of lines (line-level, not file-level)
                    perl_script = (
                        f"unless ($. >= {start_line} && $. <= {end_line}) {{ print; }}"
                    )

                cmd = ["perl", "-i", "-ne", perl_script, str(abs_target)]

            res = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
            if res.returncode != 0:
                return False, "", (
                    f"perl strategy: failed on {rel_target} hunk {h_idx + 1}: "
                    f"{res.stderr.strip() or res.stdout.strip()}"
                )

            all_stdout.append(f"perl strategy: applied hunk to {rel_target}")
            if res.stderr:
                all_stderr.append(res.stderr)

    return True, "\n".join(all_stdout), "\n".join(all_stderr)


# ---------------------------------------------------------------------------
# File snapshot / rollback helpers
# ---------------------------------------------------------------------------

def _snapshot_files(repo_root: Path, paths: Sequence[str]) -> dict[str, _FileSnapshot]:
    snapshots: dict[str, _FileSnapshot] = {}
    for rel in paths:
        abs_path = (repo_root / Path(rel)).resolve()
        if abs_path.exists():
            snapshots[rel] = _FileSnapshot(
                existed=True, content=abs_path.read_text(encoding="utf-8")
            )
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
            base_err = (
                str(result.get("error", "write_file persistence failed")).strip()
                or "write_file persistence failed"
            )
            if rollback_err:
                return False, f"{base_err}; rollback failed: {rollback_err}", "", ""
            return False, f"{base_err}; rollback applied", "", ""

    return True, "write_file persistence applied", "", ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def safe_apply_patch(
    *,
    repo_root: Path,
    diff: str,
    allowed_prefixes: Optional[Sequence[str]] = DEFAULT_ALLOWED_PREFIXES,
    check_only: bool = False,
    strategy_hint: str = "auto",
    strict_strategy: bool = False,
) -> dict[str, Any]:
    """
    Safely apply a patch inside the repository.

    Supported strategies:
      - "py"    : Python deterministic hunk application + write_file persistence
      - "sh"    : System `patch -p1` command (no git)
      - "perl"  : Perl one-liner in-place editing
      - "auto"  : Try py -> sh -> perl (best-effort chain)

    Supported patch formats:
      - Plain unified diff (--- / +++ / @@ blocks, NO git headers)
      - JSON list of file-edit operations

    Rejects any patch containing 'diff --git' headers.
    Does NOT depend on git in any way.
    """
    requested_strategy = str(strategy_hint or "auto").strip().lower() or "auto"
    strict_strategy = bool(strict_strategy)
    supported_strategies = ("auto", "py", "sh", "perl")

    if requested_strategy not in supported_strategies:
        return ApplyPatchResult(
            ok=False,
            touched_files=[],
            strategy_requested=requested_strategy,
            error=(
                f"Error: invalid strategy_hint '{requested_strategy}'. "
                f"Expected one of {supported_strategies}."
            ),
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

    # HARD REJECT: any patch containing 'diff --git' headers
    if "diff --git " in diff:
        return ApplyPatchResult(
            ok=False,
            touched_files=[],
            strategy_requested=requested_strategy,
            error=(
                "Error: 'diff --git' format is not accepted. "
                "Please provide a plain unified diff (--- / +++ / @@) "
                "or a JSON patch. " + PATCH_FORMAT_GUIDANCE
            ),
        ).to_dict()

    repo_root = repo_root.resolve()

    # Detect format
    fmt = _detect_patch_format(diff)

    if fmt == _PatchFormat.UNKNOWN:
        return ApplyPatchResult(
            ok=False,
            touched_files=[],
            strategy_requested=requested_strategy,
            error="Error: unrecognized patch format. " + PATCH_FORMAT_GUIDANCE,
        ).to_dict()

    working_diff = diff

    # Extract touched paths and detect deletions
    touched, deleted = _extract_touched_paths_and_deletes(working_diff, fmt)

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

    # Build strategy execution order
    # "auto" order: py -> sh -> perl (deterministic first, then shell tools)
    strategy_order = ["py", "sh", "perl"]
    auto_order = ["py", "sh", "perl"]

    if requested_strategy == "auto":
        run_order = list(auto_order)
    else:
        start = strategy_order.index(requested_strategy)
        run_order = [requested_strategy] if strict_strategy else strategy_order[start:]

    # Parse the patch into structured form (needed by all strategies)
    if fmt == _PatchFormat.JSON:
        parsed_files, parse_err = _parse_json_patch(working_diff)
    else:
        parsed_files, parse_err = _parse_unified_diff_strict(working_diff)

    if parse_err:
        return ApplyPatchResult(
            ok=False,
            touched_files=touched_files,
            strategy_requested=requested_strategy,
            strategy="",
            attempts=attempts,
            error=f"Error: patch parse failed: {parse_err}",
        ).to_dict()

    # Collect target paths for snapshot/rollback use
    all_target_paths: list[str] = []
    seen_targets: set[str] = set()
    for item in parsed_files:
        rel = _target_rel_path(item)
        if rel not in seen_targets:
            seen_targets.add(rel)
            all_target_paths.append(rel)

    # ---------------------------------------------------------------------------
    # Strategy 1: py (Python compute + write_file)
    # ---------------------------------------------------------------------------
    if "py" in run_order:
        py_phase = "check" if check_only else "compute"
        py_ok, computed, py_detail = _compute_python_fallback(
            repo_root=repo_root, parsed_files=parsed_files
        )
        attempts.append(_attempt("py", py_phase, py_ok, py_detail))

        if py_ok:
            if check_only:
                return ApplyPatchResult(
                    ok=True,
                    touched_files=touched_files,
                    check_only=True,
                    strategy_requested=requested_strategy,
                    strategy="py",
                    attempts=attempts,
                    stdout="py strategy check succeeded",
                ).to_dict()

            # Write files
            write_ok, write_detail, write_stdout, write_stderr = _apply_via_write_file(
                repo_root=repo_root,
                computed=computed,
                allowed_prefixes=allowed_prefixes,
                order=all_target_paths,
            )
            attempts.append(_attempt("py", "write", write_ok, write_detail))

            if write_ok:
                return ApplyPatchResult(
                    ok=True,
                    touched_files=touched_files,
                    check_only=False,
                    strategy_requested=requested_strategy,
                    strategy="py",
                    attempts=attempts,
                    stdout=write_stdout or "py strategy applied successfully",
                    stderr=write_stderr,
                ).to_dict()

            # py write failed
            if strict_strategy and requested_strategy == "py":
                return ApplyPatchResult(
                    ok=False,
                    touched_files=touched_files,
                    check_only=False,
                    strategy_requested=requested_strategy,
                    strategy="py",
                    attempts=attempts,
                    error=f"Error: py strategy write failed: {write_detail}",
                ).to_dict()

        else:
            # py compute failed
            if strict_strategy and requested_strategy == "py":
                return ApplyPatchResult(
                    ok=False,
                    touched_files=touched_files,
                    check_only=check_only,
                    strategy_requested=requested_strategy,
                    strategy="py",
                    attempts=attempts,
                    error=f"Error: py strategy failed: {py_detail}",
                ).to_dict()

    # ---------------------------------------------------------------------------
    # Strategy 2: sh (system patch command, no git)
    # ---------------------------------------------------------------------------
    if "sh" in run_order:
        sh_phase = "check" if check_only else "apply"
        sh_ok, sh_stdout, sh_stderr = _apply_via_sh(
            repo_root, parsed_files, check_only
        )
        attempts.append(
            _attempt("sh", sh_phase, sh_ok, sh_stderr if not sh_ok else sh_stdout)
        )

        if sh_ok:
            return ApplyPatchResult(
                ok=True,
                touched_files=touched_files,
                check_only=check_only,
                strategy_requested=requested_strategy,
                strategy="sh",
                attempts=attempts,
                stdout=sh_stdout,
                stderr=sh_stderr,
            ).to_dict()

        if strict_strategy and requested_strategy == "sh":
            return ApplyPatchResult(
                ok=False,
                touched_files=touched_files,
                check_only=check_only,
                strategy_requested=requested_strategy,
                strategy="sh",
                attempts=attempts,
                error=f"Error: sh strategy failed: {sh_stderr}",
            ).to_dict()

    # ---------------------------------------------------------------------------
    # Strategy 3: perl (Perl one-liner in-place editing)
    # ---------------------------------------------------------------------------
    if "perl" in run_order:
        perl_phase = "check" if check_only else "apply"

        if not check_only:
            snapshots = _snapshot_files(repo_root, all_target_paths)

        perl_ok, perl_stdout, perl_stderr = _apply_via_perl(
            repo_root, parsed_files, check_only
        )
        attempts.append(
            _attempt(
                "perl", perl_phase, perl_ok,
                perl_stderr if not perl_ok else perl_stdout,
            )
        )

        if perl_ok:
            return ApplyPatchResult(
                ok=True,
                touched_files=touched_files,
                check_only=check_only,
                strategy_requested=requested_strategy,
                strategy="perl",
                attempts=attempts,
                stdout=perl_stdout,
                stderr=perl_stderr,
            ).to_dict()

        # Rollback on failure if we modified files
        if not check_only:
            rollback_err = _rollback_files(repo_root, snapshots)  # type: ignore[possibly-undefined]
            if rollback_err:
                attempts.append(_attempt("perl", "rollback", False, rollback_err))

        if strict_strategy and requested_strategy == "perl":
            return ApplyPatchResult(
                ok=False,
                touched_files=touched_files,
                check_only=check_only,
                strategy_requested=requested_strategy,
                strategy="perl",
                attempts=attempts,
                error=f"Error: perl strategy failed: {perl_stderr}",
            ).to_dict()

    # All strategies exhausted
    return ApplyPatchResult(
        ok=False,
        touched_files=touched_files,
        check_only=check_only,
        strategy_requested=requested_strategy,
        strategy=requested_strategy if requested_strategy != "auto" else "",
        attempts=attempts,
        error="Error: all patch strategies exhausted without success.",
    ).to_dict()


# ---------------------------------------------------------------------------
# LangChain tool builder
# ---------------------------------------------------------------------------

def build_apply_patch_tool(
    *,
    repo_root: Path,
    allowed_prefixes: Optional[Sequence[str]] = DEFAULT_ALLOWED_PREFIXES,
):
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
                error=(
                    "Error: missing patch content (expected `diff` or `patch`). "
                    + PATCH_FORMAT_GUIDANCE
                ),
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
            "Safely apply a patch inside the repository. "
            "Refuses absolute paths, traversal, and paths escaping the repository root. "
            "NO-DELETE: blocks patches that delete files. "
            "Accepts plain unified diffs (--- / +++ / @@) OR JSON file-edit operations. "
            "REJECTS 'diff --git' format entirely. Does NOT use git in any way. "
            "Strategy chain: py (Python compute + write_file) -> sh (system patch -p1) -> perl (Perl one-liner). "
            "Optional strategy controls: strategy_hint=auto|py|sh|perl and strict_strategy=true|false. "
            "With check_only=true, validation runs without writing files. "
            + PATCH_FORMAT_GUIDANCE
        ),
    )
