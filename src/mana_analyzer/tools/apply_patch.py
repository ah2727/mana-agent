"""
mana_analyzer.tools.apply_patch

A safe patch-application tool for coding agents.

Key properties:
- Parses JSON patch to identify touched paths.
- Refuses patches that touch files outside repo_root.
- Optionally restricts touched paths to allowed prefixes.
- Applies patch using a deterministic strategy chain:
  py (Python compute + write_file) -> perl (Perl one-liner).
- NO-DELETE: explicitly blocks patches that delete files.
- Does NOT use, accept, or depend on git in any way.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Optional, Sequence

from .write_file import safe_write_file

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
    strategy_requested: str = "auto"
    strategy: str = ""
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _PatchLine:
    op: str   # " " for context, "+" for add, "-" for remove
    text: str


@dataclass(frozen=True)
class _PatchHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[_PatchLine]


@dataclass(frozen=True)
class _PatchFile:
    old_path: str
    new_path: str
    hunks: list[_PatchHunk]
    new_has_trailing_newline: bool = True


@dataclass(frozen=True)
class _FileSnapshot:
    existed: bool
    content: str


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    """Strip
```json ... 
``` or
``` ... 
``` wrappers that LLMs often add."""
    s = text.strip()
    if not s.startswith("```"):
        return text
    lines = s.splitlines()
    if len(lines) < 2:
        return text
    if lines[-1].strip() != "```":
        return text
    inner = "\n".join(lines[1:-1])
    return inner.strip() + "\n"


def _looks_like_git_diff_payload(text: str) -> bool:
    """Detect likely git/unified-diff payloads so we can fail fast with guidance."""
    s = (text or "").lstrip()
    if not s:
        return False

    if s.startswith("diff --git "):
        return True

    lines = [line for line in s.splitlines() if line.strip()]
    if not lines:
        return False

    head = lines[:12]
    has_file_headers = any(line.startswith("--- ") for line in head) and any(
        line.startswith("+++ ") for line in head
    )
    has_hunk_header = any(line.startswith("@@ ") for line in lines[:80])
    return has_file_headers and has_hunk_header


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


def _is_dev_null(path: str) -> bool:
    return path in {"dev/null", "/dev/null"}


# ---------------------------------------------------------------------------
# Path extraction and validation
# ---------------------------------------------------------------------------

def _extract_touched_paths_and_deletes(
    patch_data: list[dict[str, Any]],
) -> tuple[set[str], set[str]]:
    """Extract touched file paths and any delete markers from parsed JSON data."""
    touched: set[str] = set()
    deleted: set[str] = set()

    for entry in patch_data:
        path = _normalise_user_path(str(entry.get("path", "")))
        if path:
            touched.add(path)
        if entry.get("delete", False):
            deleted.add(path)

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

        if raw.startswith("/"):
            return False, [], f"Blocked: absolute path in patch: {p}"

        parts = [
            seg for seg in raw.replace("\\", "/").split("/")
            if seg not in ("", ".")
        ]
        if any(seg == ".." for seg in parts):
            return False, [], f"Blocked: traversal ('..') in patch path: {p}"

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
# JSON patch parser
# ---------------------------------------------------------------------------

def _parse_json_patch_data(
    data: list[dict[str, Any]],
) -> tuple[list[_PatchFile], str]:
    """
    Parse already-deserialised JSON patch data into _PatchFile structures.

    Expected element schema::

        {
            "path": "src/foo.py",
            "create": false,          # optional, default false
            "hunks": [
                {
                    "old_start": 10,
                    "old_lines": ["line to remove or context"],
                    "new_lines": ["replacement line"]
                }
            ]
        }
    """
    if not isinstance(data, list):
        return [], "Error: JSON patch must be a list of file-edit objects"

    parsed: list[_PatchFile] = []

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

        hunks: list[_PatchHunk] = []

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

            patch_lines: list[_PatchLine] = []
            for ol in old_lines:
                patch_lines.append(_PatchLine(op="-", text=ol))
            for nl in new_lines:
                patch_lines.append(_PatchLine(op="+", text=nl))

            old_count = len(old_lines)
            new_count = len(new_lines)

            hunks.append(
                _PatchHunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=old_start,
                    new_count=new_count,
                    lines=patch_lines,
                )
            )

        old_path = "/dev/null" if is_create else path
        new_path = path

        parsed.append(
            _PatchFile(
                old_path=old_path,
                new_path=new_path,
                hunks=hunks,
                new_has_trailing_newline=True,
            )
        )

    if not parsed:
        return [], "Error: no valid entries in JSON patch"

    return parsed, ""


def _parse_json_patch(text: str) -> tuple[list[dict[str, Any]], list[_PatchFile], str]:
    """
    Deserialise a JSON string, return (raw_data, parsed_files, error).

    Convenience wrapper that does JSON decode + structural parse in one call.
    """
    try:
        data = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError) as exc:
        return [], [], f"Error: invalid JSON: {exc}"

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        return [], [], "Error: JSON patch must be a list of file-edit objects"

    parsed, err = _parse_json_patch_data(data)
    return data, parsed, err


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
# Common helpers
# ---------------------------------------------------------------------------

def _target_rel_path(pf: _PatchFile) -> str:
    if not _is_dev_null(pf.new_path):
        return _normalise_user_path(pf.new_path)
    return _normalise_user_path(pf.old_path)


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
    hunks: Sequence[_PatchHunk],
    file_path: str,
) -> tuple[bool, list[str], str]:
    """Apply a sequence of hunks to *base_lines* and return the result."""
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
                        f"hunk {idx}: context mismatch at line {cursor + 1} "
                        f"for {file_path}. "
                        f"Expected {line.text!r}, got {out[cursor]!r}"
                    )
                cursor += 1

        replacement = [
            line.text for line in hunk.lines if line.op in {" ", "+"}
        ]
        out[pos : pos + hunk.old_count] = replacement
        delta += len(replacement) - hunk.old_count

    return True, out, ""


# ---------------------------------------------------------------------------
# File snapshot / rollback helpers
# ---------------------------------------------------------------------------

def _snapshot_files(
    repo_root: Path, paths: Sequence[str],
) -> dict[str, _FileSnapshot]:
    snapshots: dict[str, _FileSnapshot] = {}
    for rel in paths:
        abs_path = (repo_root / Path(rel)).resolve()
        if abs_path.exists():
            snapshots[rel] = _FileSnapshot(
                existed=True,
                content=abs_path.read_text(encoding="utf-8"),
            )
        else:
            snapshots[rel] = _FileSnapshot(existed=False, content="")
    return snapshots


def _rollback_files(
    repo_root: Path, snapshots: dict[str, _FileSnapshot],
) -> str:
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
                return (
                    False,
                    f"{base_err}; rollback failed: {rollback_err}",
                    "",
                    "",
                )
            return False, f"{base_err}; rollback applied", "", ""

    return True, "write_file persistence applied", "", ""


# ---------------------------------------------------------------------------
# Strategy: py  (Python deterministic compute + write_file persistence)
# ---------------------------------------------------------------------------

def _compute_python(
    *,
    repo_root: Path,
    parsed_files: Sequence[_PatchFile],
) -> tuple[bool, dict[str, str], str]:
    computed: dict[str, str] = {}
    newline_prefs: dict[str, bool] = {}

    for pf in parsed_files:
        rel_target = _target_rel_path(pf)
        abs_target = (repo_root / Path(rel_target)).resolve()

        if rel_target in computed:
            base_lines = computed[rel_target].splitlines()
        elif _is_dev_null(pf.old_path):
            base_lines: list[str] = []
        else:
            if not abs_target.exists():
                return False, {}, f"py strategy: missing target file {rel_target}"
            base_lines = abs_target.read_text(encoding="utf-8").splitlines()

        ok, patched_lines, err = _apply_hunks_to_lines(
            base_lines=base_lines,
            hunks=pf.hunks,
            file_path=rel_target,
        )
        if not ok:
            return False, {}, f"py strategy failed: {err}"

        newline_prefs[rel_target] = bool(pf.new_has_trailing_newline)
        computed[rel_target] = _lines_to_text(
            patched_lines,
            trailing_newline=newline_prefs[rel_target],
        )

    return True, computed, "py strategy computed updates"


# ---------------------------------------------------------------------------
# Strategy: perl  (Perl one-liner in-place editing)
# ---------------------------------------------------------------------------

def _apply_via_perl(
    repo_root: Path,
    parsed_files: list[_PatchFile],
    check_only: bool,
) -> tuple[bool, str, str]:
    """Apply using ``perl -i`` in-place editing, one hunk at a time."""
    try:
        perl_check = subprocess.run(
            ["perl", "-v"], capture_output=True, text=True,
        )
        if perl_check.returncode != 0:
            return False, "", "perl strategy: perl not available"
    except FileNotFoundError:
        return False, "", "perl strategy: perl not found on system"

    if check_only:
        return (
            True,
            "perl strategy: check passed (perl available, patch parseable)",
            "",
        )

    all_stdout: list[str] = []
    all_stderr: list[str] = []

    for pf in parsed_files:
        rel_target = _target_rel_path(pf)
        abs_target = (repo_root / Path(rel_target)).resolve()

        # New-file creation — collect added lines and write directly.
        if _is_dev_null(pf.old_path):
            new_content_lines: list[str] = []
            for hunk in pf.hunks:
                for pl in hunk.lines:
                    if pl.op == "+":
                        new_content_lines.append(pl.text)
            content = _lines_to_text(
                new_content_lines,
                trailing_newline=pf.new_has_trailing_newline,
            )
            abs_target.parent.mkdir(parents=True, exist_ok=True)
            abs_target.write_text(content, encoding="utf-8")
            all_stdout.append(f"perl strategy: created {rel_target}")
            continue

        if not abs_target.exists():
            return False, "", f"perl strategy: missing target file {rel_target}"

        # Process hunks in reverse order to keep line numbers stable.
        sorted_hunks = sorted(
            pf.hunks, key=lambda h: h.old_start, reverse=True,
        )

        for h_idx, hunk in enumerate(sorted_hunks):
            new_lines_text = [
                pl.text for pl in hunk.lines if pl.op in {" ", "+"}
            ]

            start_line = hunk.old_start
            end_line = hunk.old_start + hunk.old_count - 1

            if hunk.old_count == 0:
                # Pure insertion
                insert_text = "\n".join(new_lines_text)
                insert_escaped = (
                    insert_text.replace("\\", "\\\\").replace("'", "\\'")
                )
                if start_line <= 1:
                    perl_script = (
                        f"BEGIN {{ print '{insert_escaped}\\n' }}"
                    )
                else:
                    target_line = start_line - 1
                    perl_script = (
                        f"if ($. == {target_line}) {{ print; "
                        f"print '{insert_escaped}\\n'; next }}"
                    )
                cmd = ["perl", "-i", "-pe", perl_script, str(abs_target)]
            else:
                new_text = "\n".join(new_lines_text)
                new_escaped = (
                    new_text.replace("\\", "\\\\").replace("'", "\\'")
                )

                if new_lines_text:
                    # Replacement
                    perl_script = (
                        f"if ($. == {start_line}) "
                        f"{{ print '{new_escaped}\\n'; }} "
                        f"elsif ($. > {start_line} && $. <= {end_line}) "
                        f"{{ }} "
                        f"else {{ print; }}"
                    )
                else:
                    # Pure removal (line-level, not file-level)
                    perl_script = (
                        f"unless ($. >= {start_line} && $. <= {end_line}) "
                        f"{{ print; }}"
                    )

                cmd = ["perl", "-i", "-ne", perl_script, str(abs_target)]

            res = subprocess.run(
                cmd, cwd=repo_root, capture_output=True, text=True,
            )
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
# Main entry point
# ---------------------------------------------------------------------------

def safe_apply_patch(
    *,
    repo_root: Path,
    patch: str,
    allowed_prefixes: Optional[Sequence[str]] = DEFAULT_ALLOWED_PREFIXES,
    check_only: bool = False,
    strategy_hint: str = "auto",
    strict_strategy: bool = False,
) -> dict[str, Any]:
    """
    Safely apply a JSON patch inside the repository.

    Supported strategies
    --------------------
    - ``"py"``      — Python deterministic hunk application + write_file
    - ``"perl"``    — Perl one-liner in-place editing
    - ``"auto"``    — Try py → perl (best-effort chain)

    Patch format
    ------------
    JSON list of file-edit operations::

        [
          {
            "path": "src/foo.py",
            "create": false,
            "hunks": [
              {
                "old_start": 10,
                "old_lines": ["old line"],
                "new_lines": ["new line"]
              }
            ]
          }
        ]

    Does NOT depend on git in any way.
    """
    requested_strategy = (
        str(strategy_hint or "auto").strip().lower() or "auto"
    )
    strict_strategy = bool(strict_strategy)
    supported_strategies = ("auto", "py", "perl")

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

    if not patch.strip():
        return ApplyPatchResult(
            ok=False,
            touched_files=[],
            strategy_requested=requested_strategy,
            error="Error: empty patch content",
        ).to_dict()

    # Strip markdown fences that LLMs sometimes wrap around JSON.
    patch = _strip_markdown_fences(patch)

    if _looks_like_git_diff_payload(patch):
        return ApplyPatchResult(
            ok=False,
            touched_files=[],
            strategy_requested=requested_strategy,
            error=(
                "Error: git diff/unified diff payloads are not supported. "
                "Use JSON patch list format, e.g. "
                "[{\"path\":\"src/foo.py\",\"create\":false,"
                "\"hunks\":[{\"old_start\":1,\"old_lines\":[\"old\"],"
                "\"new_lines\":[\"new\"]}]}]."
            ),
        ).to_dict()

    repo_root = repo_root.resolve()

    # ---- Parse JSON once, derive both raw data and structured files --------
    raw_data, parsed_files, parse_err = _parse_json_patch(patch)

    if parse_err:
        return ApplyPatchResult(
            ok=False,
            touched_files=[],
            strategy_requested=requested_strategy,
            error=(
                "Error: patch parse failed: "
                f"{parse_err}. Expected JSON patch list with keys "
                "path/create/hunks(old_start, old_lines, new_lines)."
            ),
        ).to_dict()

    # ---- Extract touched paths / detect deletions --------------------------
    touched, deleted = _extract_touched_paths_and_deletes(raw_data)

    if deleted:
        return ApplyPatchResult(
            ok=False,
            touched_files=sorted(touched),
            strategy_requested=requested_strategy,
            error=(
                f"Blocked: patch deletes files (not allowed): {sorted(deleted)}"
            ),
        ).to_dict()

    ok, touched_files, err = _validate_touched_paths(
        repo_root, touched, allowed_prefixes,
    )
    if not ok:
        return ApplyPatchResult(
            ok=False,
            touched_files=[],
            strategy_requested=requested_strategy,
            error=err,
        ).to_dict()

    # ---- Collect target paths for snapshots --------------------------------
    all_target_paths: list[str] = []
    seen_targets: set[str] = set()
    for item in parsed_files:
        rel = _target_rel_path(item)
        if rel not in seen_targets:
            seen_targets.add(rel)
            all_target_paths.append(rel)

    # ---- Build strategy run-order ------------------------------------------
    strategy_order = ["py", "perl"]
    attempts: list[dict[str, Any]] = []

    if requested_strategy == "auto":
        run_order = list(strategy_order)
    else:
        start = strategy_order.index(requested_strategy)
        run_order = (
            [requested_strategy] if strict_strategy else strategy_order[start:]
        )

    # -----------------------------------------------------------------------
    # Strategy: py
    # -----------------------------------------------------------------------
    if "py" in run_order:
        py_phase = "check" if check_only else "compute"
        py_ok, computed, py_detail = _compute_python(
            repo_root=repo_root, parsed_files=parsed_files,
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

            write_ok, write_detail, write_stdout, write_stderr = (
                _apply_via_write_file(
                    repo_root=repo_root,
                    computed=computed,
                    allowed_prefixes=allowed_prefixes,
                    order=all_target_paths,
                )
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

    # -----------------------------------------------------------------------
    # Strategy: perl
    # -----------------------------------------------------------------------
    if "perl" in run_order:
        perl_phase = "check" if check_only else "apply"

        snapshots: dict[str, _FileSnapshot] = {}
        if not check_only:
            snapshots = _snapshot_files(repo_root, all_target_paths)

        perl_ok, perl_stdout, perl_stderr = _apply_via_perl(
            repo_root, parsed_files, check_only,
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

        if not check_only and snapshots:
            rollback_err = _rollback_files(repo_root, snapshots)
            if rollback_err:
                attempts.append(
                    _attempt("perl", "rollback", False, rollback_err),
                )

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
        strategy=(
            requested_strategy if requested_strategy != "auto" else ""
        ),
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
    """Build a LangChain StructuredTool that wraps :func:`safe_apply_patch`."""
    try:
        from langchain_core.tools import StructuredTool  # type: ignore[import-untyped]
    except Exception:  # pragma: no cover
        from langchain.tools import StructuredTool  # type: ignore[import-untyped]

    def _tool(
        patch: str | None = None,
        check_only: bool = False,
        strategy_hint: str = "auto",
        strict_strategy: bool = False,
    ) -> dict[str, Any]:
        if patch is None:
            return ApplyPatchResult(
                ok=False,
                touched_files=[],
                check_only=check_only,
                strategy_requested=str(strategy_hint or "auto"),
                error="Error: missing patch content (expected `patch` parameter).",
            ).to_dict()
        return safe_apply_patch(
            repo_root=repo_root,
            patch=patch,
            allowed_prefixes=allowed_prefixes,
            check_only=check_only,
            strategy_hint=str(strategy_hint or "auto"),
            strict_strategy=bool(strict_strategy),
        )

    return StructuredTool.from_function(
        func=_tool,
        name="apply_patch",
        description=(
            "Safely apply a JSON patch inside the repository. "
            "Refuses absolute paths, traversal, and paths escaping the repo root. "
            "NO-DELETE: blocks patches that delete files. "
            "Accepts JSON file-edit operations only. "
            "Does NOT use git in any way. "
            "Strategy chain: py -> perl. "
            "Controls: strategy_hint=auto|py|perl, "
            "strict_strategy=true|false. "
            "With check_only=true, validation runs without writing files."
        ),
    )
