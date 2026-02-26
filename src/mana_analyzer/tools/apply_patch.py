"""
mana_analyzer.tools.apply_patch

A safe patch-application tool for coding agents.

Key properties:
- Parses the patch to identify touched paths.
- Refuses patches that touch files outside repo_root.
- Optionally restricts touched paths to allowed prefixes.
- Applies patch using a deterministic strategy chain:
  git apply -> perl fallback -> python fallback (compute) -> write_file persistence.
- Intended usage: run patch validation (`check_only=true`) first, then apply.
- NO-DELETE: explicitly blocks patches that delete files (/dev/null targets).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Optional, Sequence

from .write_file import safe_write_file

logger = logging.getLogger(__name__)

# None => allow any path under repo_root
DEFAULT_ALLOWED_PREFIXES: Optional[tuple[str, ...]] = None

_DRIVE_LETTER_RE = re.compile(r"^[a-zA-Z]:[\\/]")
_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


# Guidance to reduce common "not a git patch" failures.
GIT_UNIFIED_DIFF_GUIDANCE = (
    "Patch must be a VALID unified diff accepted by `git apply`, e.g.:\n"
    "  diff --git a/path/to/file.py b/path/to/file.py\n"
    "  --- a/path/to/file.py\n"
    "  +++ b/path/to/file.py\n"
    "  @@ -1,3 +1,4 @@\n"
    "Do NOT use '*** Begin Patch'/'*** End Patch' format. "
    "Do NOT use absolute paths or '..'."
)

_PERL_FALLBACK_SCRIPT = """
use strict;
use warnings;
use JSON::PP qw(decode_json encode_json);

sub fail {
  my ($msg) = @_;
  print encode_json({ ok => JSON::PP::false(), detail => $msg });
  exit 0;
}

my ($repo_root, $plan_json, $check_only_flag) = @ARGV;
if (!defined $repo_root || !defined $plan_json || !defined $check_only_flag) {
  fail("invalid arguments");
}
my $check_only = $check_only_flag ? 1 : 0;

my $plan = eval { decode_json($plan_json) };
if ($@ || ref($plan) ne 'HASH') {
  fail("invalid plan json");
}

my $files = $plan->{files};
if (ref($files) ne 'ARRAY') {
  fail("plan missing files");
}

for my $f (@$files) {
  if (ref($f) ne 'HASH') {
    fail("invalid file entry");
  }

  my $old_path = $f->{old_path};
  my $new_path = $f->{new_path};
  my $new_has_trailing_newline = exists $f->{new_has_trailing_newline} ? ($f->{new_has_trailing_newline} ? 1 : 0) : 1;
  my $target_rel = ($new_path eq 'dev/null' || $new_path eq '/dev/null') ? $old_path : $new_path;
  my $target_abs = "$repo_root/$target_rel";

  my @base_lines = ();
  if (!($old_path eq 'dev/null' || $old_path eq '/dev/null')) {
    if (!-e $target_abs) {
      fail("missing target file: $target_rel");
    }
    open my $rfh, '<:encoding(UTF-8)', $target_abs or fail("unable to read $target_rel: $!");
    my @raw = <$rfh>;
    close $rfh;
    chomp @raw;
    @base_lines = @raw;
  }

  my @lines = @base_lines;
  my $offset = 0;
  my $hunks = $f->{hunks};
  if (ref($hunks) ne 'ARRAY') {
    fail("invalid hunks for $target_rel");
  }

  my $hidx = 0;
  for my $h (@$hunks) {
    $hidx += 1;
    my $old_start = int($h->{old_start});
    my $old_count = int($h->{old_count});
    my $hunk_lines = $h->{lines};
    if (ref($hunk_lines) ne 'ARRAY') {
      fail("invalid hunk lines for $target_rel");
    }

    my $expected = $old_start - 1 + $offset;
    if ($expected < 0 || $expected > scalar(@lines)) {
      fail("hunk $hidx out of range for $target_rel");
    }

    my $cursor = $expected;
    for my $ln (@$hunk_lines) {
      my $op = $ln->{op};
      my $text = defined($ln->{text}) ? $ln->{text} : '';
      if ($op eq ' ' || $op eq '-') {
        if ($cursor >= scalar(@lines)) {
          fail("hunk $hidx context mismatch at EOF for $target_rel");
        }
        if ($lines[$cursor] ne $text) {
          fail("hunk $hidx context mismatch at line " . ($cursor + 1) . " for $target_rel");
        }
        $cursor += 1;
      }
    }

    my @replacement = ();
    for my $ln (@$hunk_lines) {
      my $op = $ln->{op};
      if ($op eq ' ' || $op eq '+') {
        push @replacement, (defined($ln->{text}) ? $ln->{text} : '');
      }
    }

    splice(@lines, $expected, $old_count, @replacement);
    $offset += scalar(@replacement) - $old_count;
  }

  if (!$check_only) {
    open my $wfh, '>:encoding(UTF-8)', $target_abs or fail("unable to write $target_rel: $!");
    if (scalar(@lines) > 0) {
      if ($new_has_trailing_newline) {
        print {$wfh} join("\n", @lines), "\n";
      } else {
        print {$wfh} join("\n", @lines);
      }
    }
    close $wfh;
  }
}

print encode_json({ ok => JSON::PP::true(), detail => "perl fallback applied" });
"""


@dataclass(frozen=True)
class ApplyPatchResult:
    ok: bool
    touched_files: list[str]
    strip_level: int = -1
    check_only: bool = False
    stdout: str = ""
    stderr: str = ""
    error: str = ""
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
    If an LLM wraps diffs in ```diff ... ``` or ``` ... ```, strip the fences.
    This keeps the tool tolerant without changing the semantics of the patch.
    """
    s = text.strip()
    if not s.startswith("```"):
        return text
    lines = s.splitlines()
    if len(lines) < 2:
        return text
    if not lines[-1].strip().startswith("```"):
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
            if not line.strip():
                i += 1
                continue
            return [], f"Error: unsupported prelude line: {line[:120]}"

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
                old_count = int(m.group(2) or "1")
                new_start = int(m.group(3))
                new_count = int(m.group(4) or "1")
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
                        return [], "Error: malformed hunk line"
                    op = hline[0]
                    if op not in {" ", "+", "-"}:
                        return [], f"Error: unsupported hunk line prefix: {op}"
                    hunk_lines.append(_DiffLine(op=op, text=hline[1:]))
                    last_hunk_op = op
                    i += 1

                old_seen = sum(1 for x in hunk_lines if x.op in {" ", "-"})
                new_seen = sum(1 for x in hunk_lines if x.op in {" ", "+"})
                if old_seen != old_count or new_seen != new_count:
                    return [], "Error: hunk line counts do not match header"

                hunks.append(
                    _DiffHunk(
                        old_start=old_start,
                        old_count=old_count,
                        new_start=new_start,
                        new_count=new_count,
                        lines=hunk_lines,
                    )
                )
                continue
            if not current.strip():
                i += 1
                continue
            return [], f"Error: unsupported diff metadata line: {current[:120]}"

        if not saw_old or not saw_new:
            return [], "Error: malformed diff block (missing ---/+++ headers)"
        if not hunks:
            if saw_mode_metadata:
                return [], "Error: unsupported diff feature (mode-only diff)"
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


def _run(cmd: list[str], *, cwd: Path, stdin_text: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        input=stdin_text,
        text=True,
        capture_output=True,
        cwd=str(cwd),
        check=False,
    )


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
                    return False, out, f"hunk {idx}: context mismatch at line {cursor + 1} for {file_path}"
                cursor += 1

        replacement = [line.text for line in hunk.lines if line.op in {" ", "+"}]
        out[pos : pos + hunk.old_count] = replacement
        delta += len(replacement) - hunk.old_count

    return True, out, ""


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


def _perl_plan_payload(parsed_files: Sequence[_DiffFile]) -> str:
    payload = {
        "files": [
            {
                "old_path": item.old_path,
                "new_path": item.new_path,
                "new_has_trailing_newline": bool(item.new_has_trailing_newline),
                "hunks": [
                    {
                        "old_start": h.old_start,
                        "old_count": h.old_count,
                        "new_start": h.new_start,
                        "new_count": h.new_count,
                        "lines": [{"op": line.op, "text": line.text} for line in h.lines],
                    }
                    for h in item.hunks
                ],
            }
            for item in parsed_files
        ]
    }
    return json.dumps(payload, separators=(",", ":"))


def _run_perl_fallback(
    *,
    repo_root: Path,
    parsed_files: Sequence[_DiffFile],
    check_only: bool,
) -> tuple[bool, str, str, str]:
    plan_json = _perl_plan_payload(parsed_files)
    proc = subprocess.run(
        ["perl", "-", "--", str(repo_root), plan_json, "1" if check_only else "0"],
        input=_PERL_FALLBACK_SCRIPT,
        text=True,
        capture_output=True,
        check=False,
    )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    if proc.returncode != 0:
        return False, f"perl fallback exited with code {proc.returncode}", stdout, stderr

    detail = "perl fallback failed"
    try:
        payload = json.loads(stdout.strip() or "{}")
        ok = bool(payload.get("ok"))
        detail = str(payload.get("detail", "")).strip() or detail
        return ok, detail, stdout, stderr
    except json.JSONDecodeError:
        detail = (stdout.strip() or stderr.strip() or detail)[:400]
        return False, detail, stdout, stderr


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
) -> dict[str, Any]:
    if not diff.strip():
        return ApplyPatchResult(ok=False, touched_files=[], error="Error: empty diff").to_dict()

    # Common LLM behavior: wrap in ```diff fences.
    diff = _strip_markdown_fences(diff)

    # Reject non-git patch formats early with a clear error.
    # (Example: "*** Begin Patch" format used by some patch tools, not git apply.)
    head = diff.lstrip()[:200]
    if head.startswith("*** Begin Patch") or "*** Begin Patch" in head:
        return ApplyPatchResult(
            ok=False,
            touched_files=[],
            error="Error: patch is not a git-unified diff. " + GIT_UNIFIED_DIFF_GUIDANCE,
        ).to_dict()

    # Another early signal: if it doesn't mention diff --git anywhere, it's very likely invalid.
    if "diff --git " not in diff:
        return ApplyPatchResult(
            ok=False,
            touched_files=[],
            error="Error: patch missing 'diff --git' header. " + GIT_UNIFIED_DIFF_GUIDANCE,
        ).to_dict()

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

    parsed_files, parse_err = _parse_unified_diff_strict(diff)
    if parse_err:
        return ApplyPatchResult(ok=False, touched_files=touched_files, error=parse_err).to_dict()

    attempts: list[dict[str, Any]] = []

    # Strategy 1: git apply (preserve -p0/-p1/-p2 behavior)
    if shutil.which("git") is None:
        attempts.append(_attempt("git", "check", False, "git not found on PATH"))
    else:
        last_stderr = ""
        for p in (0, 1, 2):
            check_cmd = ["git", "apply", f"-p{p}", "--check", "-"]
            proc = _run(check_cmd, cwd=repo_root, stdin_text=diff)
            check_ok = proc.returncode == 0
            attempts.append(_attempt("git", f"check-p{p}", check_ok, proc.stderr or "ok"))
            if not check_ok:
                last_stderr = proc.stderr or last_stderr
                continue

            if check_only:
                return ApplyPatchResult(
                    ok=True,
                    touched_files=touched_files,
                    strip_level=p,
                    check_only=True,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    strategy="git",
                    attempts=attempts,
                ).to_dict()

            apply_cmd = ["git", "apply", f"-p{p}", "--whitespace=nowarn", "-"]
            proc2 = _run(apply_cmd, cwd=repo_root, stdin_text=diff)
            apply_ok = proc2.returncode == 0
            attempts.append(_attempt("git", f"apply-p{p}", apply_ok, proc2.stderr or "ok"))
            if apply_ok:
                logger.info("Applied patch with git -p%d touching %d files", p, len(touched_files))
                return ApplyPatchResult(
                    ok=True,
                    touched_files=touched_files,
                    strip_level=p,
                    check_only=False,
                    stdout=proc2.stdout,
                    stderr=proc2.stderr,
                    strategy="git",
                    attempts=attempts,
                ).to_dict()

            # If git check passed but git apply failed, move to next strategy.
            last_stderr = proc2.stderr or last_stderr
            break

        if last_stderr:
            attempts.append(_attempt("git", "summary", False, last_stderr.strip()[:400]))

    # Strategy 2: perl fallback applier
    perl_phase = "check" if check_only else "apply"
    if shutil.which("perl") is None:
        attempts.append(_attempt("perl", perl_phase, False, "perl not found on PATH"))
    else:
        perl_snapshots: dict[str, _FileSnapshot] | None = None
        if not check_only:
            perl_order: list[str] = []
            perl_seen: set[str] = set()
            for item in parsed_files:
                rel = _target_rel_path(item)
                if rel in perl_seen:
                    continue
                perl_seen.add(rel)
                perl_order.append(rel)
            perl_snapshots = _snapshot_files(repo_root, perl_order)
        perl_ok, perl_detail, perl_stdout, perl_stderr = _run_perl_fallback(
            repo_root=repo_root,
            parsed_files=parsed_files,
            check_only=check_only,
        )
        if (not check_only) and (not perl_ok) and perl_snapshots is not None:
            rollback_err = _rollback_files(repo_root, perl_snapshots)
            if rollback_err:
                perl_detail = f"{perl_detail}; rollback failed: {rollback_err}"
            else:
                perl_detail = f"{perl_detail}; rollback applied"
        attempts.append(_attempt("perl", perl_phase, perl_ok, perl_detail))
        if perl_ok:
            return ApplyPatchResult(
                ok=True,
                touched_files=touched_files,
                check_only=check_only,
                stdout=perl_stdout,
                stderr=perl_stderr,
                strategy="perl",
                attempts=attempts,
            ).to_dict()

    # Strategy 3: python fallback compute-only
    python_phase = "check" if check_only else "compute"
    py_ok, computed, py_detail = _compute_python_fallback(repo_root=repo_root, parsed_files=parsed_files)
    attempts.append(_attempt("python", python_phase, py_ok, py_detail))

    if not py_ok:
        return ApplyPatchResult(
            ok=False,
            touched_files=touched_files,
            check_only=check_only,
            strategy="python",
            attempts=attempts,
            error=f"Error: patch fallback chain exhausted. {py_detail}",
        ).to_dict()

    if check_only:
        return ApplyPatchResult(
            ok=True,
            touched_files=touched_files,
            check_only=True,
            strategy="python",
            attempts=attempts,
            stdout="python fallback check succeeded",
        ).to_dict()

    # Strategy 4: write_file transactional persistence
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
            strategy="write_file",
            attempts=attempts,
            stdout=write_stdout,
            stderr=write_stderr,
        ).to_dict()

    return ApplyPatchResult(
        ok=False,
        touched_files=touched_files,
        check_only=False,
        strategy="write_file",
        attempts=attempts,
        error=f"Error: patch fallback chain exhausted. {write_detail}",
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
                error="Error: missing patch content (expected `diff` or `patch`). " + GIT_UNIFIED_DIFF_GUIDANCE,
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
            "Internal fallback chain: git apply -> perl fallback -> python fallback compute -> write_file persistence. "
            "With check_only=true, validation runs through git/perl/python without writing files. "
            + GIT_UNIFIED_DIFF_GUIDANCE
        ),
    )
