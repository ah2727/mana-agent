#!/usr/bin/env python3
"""Generate a filled Pull Request body from branch commits and file changes.

Used by ``.github/workflows/pr-autofill.yml`` so new PRs are not left as empty
template scaffolding. Static GitHub PR templates cannot fill themselves; this
script supplies the missing dynamic content from git history and the diff.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path


TEMPLATE_MARKER = "<!-- mana-agent:pr-template -->"
AUTOFILLED_MARKER = "<!-- mana-agent:pr-autofilled -->"

# Paths → type of change hints
PATH_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^(\.github/|scripts/mana_agent_entry\.py)"), "ci"),
    (re.compile(r"^(docs/|README\.md|CONTRIBUTING\.md|CODE_OF_CONDUCT\.md|SECURITY\.md)"), "docs"),
    (re.compile(r"^tests/"), "tests"),
    (re.compile(r"(^|/)CHANGELOG\.md$"), "docs"),
    (re.compile(r"pyproject\.toml|requirements"), "tooling"),
]

COMMIT_TYPE_MAP = {
    "feat": "feature",
    "feature": "feature",
    "fix": "bug",
    "bug": "bug",
    "bugfix": "bug",
    "docs": "docs",
    "doc": "docs",
    "refactor": "improvement",
    "perf": "improvement",
    "improve": "improvement",
    "enhancement": "improvement",
    "test": "tests",
    "tests": "tests",
    "ci": "ci",
    "build": "ci",
    "chore": "ci",
    "release": "ci",
    "security": "security",
    "sec": "security",
    "break": "breaking",
    "breaking": "breaking",
}


def run_git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout.strip() if result.returncode == 0 else ""


def commit_subjects(base: str, head: str) -> list[str]:
    log = run_git("log", "--format=%s", f"{base}..{head}", check=False)
    if not log:
        return []
    return [line.strip() for line in log.splitlines() if line.strip()]


def changed_files(base: str, head: str) -> list[str]:
    out = run_git("diff", "--name-only", f"{base}...{head}", check=False)
    if not out:
        out = run_git("diff", "--name-only", f"{base}..{head}", check=False)
    return [line.strip() for line in out.splitlines() if line.strip()]


def infer_types(subjects: list[str], files: list[str]) -> set[str]:
    types: set[str] = set()
    for subject in subjects:
        lower = subject.lower()
        if re.search(r"\bbreaking\b|!:", lower):
            types.add("breaking")
        conv = re.match(r"^(\w+)(\(.+\))?!?:\s*", subject)
        if conv:
            mapped = COMMIT_TYPE_MAP.get(conv.group(1).lower())
            if mapped:
                types.add(mapped)
                continue
        for key, mapped in COMMIT_TYPE_MAP.items():
            if lower.startswith(f"{key}:") or lower.startswith(f"{key}("):
                types.add(mapped)
                break
        if "fix" in lower and "bug" not in types and "feature" not in types:
            # weak signal only if no stronger type found later
            pass
        if re.search(r"\bfix(es|ed)?\b|\bbug\b", lower):
            types.add("bug")
        if re.search(r"\bfeat(ure)?s?\b|\badd(s|ed)?\b", lower):
            types.add("feature")

    path_counter: Counter[str] = Counter()
    for path in files:
        matched = False
        for pattern, hint in PATH_HINTS:
            if pattern.search(path):
                path_counter[hint] += 1
                matched = True
                break
        if not matched:
            if path.startswith("src/"):
                path_counter["code"] += 1
            else:
                path_counter["other"] += 1

    total = sum(path_counter.values()) or 1
    docs_only = path_counter.get("docs", 0) == total
    tests_only = path_counter.get("tests", 0) == total
    ci_only = path_counter.get("ci", 0) + path_counter.get("tooling", 0) == total

    if docs_only:
        types.add("docs")
    if tests_only and not any(t in types for t in ("bug", "feature", "improvement", "breaking")):
        types.add("tests")
    if ci_only and not any(
        t in types for t in ("bug", "feature", "improvement", "breaking", "docs")
    ):
        types.add("ci")
    if "security" in " ".join(subjects).lower() or any("security" in f.lower() for f in files):
        types.add("security")

    if not types:
        if path_counter.get("docs") and path_counter.get("code", 0) == 0:
            types.add("docs")
        elif path_counter.get("ci") and path_counter.get("code", 0) == 0:
            types.add("ci")
        elif path_counter.get("tests") and path_counter.get("code", 0) == 0:
            types.add("tests")
        else:
            types.add("improvement")
    return types


def checkbox(label: str, checked: bool) -> str:
    mark = "x" if checked else " "
    return f"- [{mark}] {label}"


def format_type_section(types: set[str]) -> str:
    mapping = [
        ("bug", "Bug fix"),
        ("feature", "New feature"),
        ("improvement", "Improvement / refactor"),
        ("breaking", "Breaking change"),
        ("docs", "Documentation"),
        ("tests", "Tests only"),
        ("ci", "CI / tooling / release"),
        ("security", "Security-related"),
    ]
    return "\n".join(checkbox(label, key in types) for key, label in mapping)


def summarize(subjects: list[str], files: list[str]) -> str:
    if subjects:
        # Prefer non-merge subjects
        usable = [s for s in subjects if not s.lower().startswith("merge ")]
        if not usable:
            usable = subjects
        primary = usable[0]
        primary = re.sub(r"^(\w+)(\(.+\))?!?:\s*", "", primary).strip()
        if len(usable) == 1:
            return primary.rstrip(".") + "."
        return f"{primary.rstrip('.')} (+{len(usable) - 1} more commit{'s' if len(usable) != 2 else ''})."
    if files:
        return f"Update {len(files)} file{'s' if len(files) != 1 else ''} on this branch."
    return "Branch changes relative to the base ref."


def changes_list(subjects: list[str], files: list[str], limit: int = 12) -> list[str]:
    items: list[str] = []
    for subject in subjects:
        if subject.lower().startswith("merge "):
            continue
        cleaned = re.sub(r"^(\w+)(\(.+\))?!?:\s*", "", subject).strip()
        if cleaned and cleaned not in items:
            items.append(cleaned)
        if len(items) >= limit:
            break
    if not items and files:
        for path in files[:limit]:
            items.append(f"`{path}`")
    if not items:
        items.append("See commits on this branch.")
    return items


def group_files(files: list[str]) -> list[str]:
    if not files:
        return ["- _(no file list available)_"]
    buckets: dict[str, list[str]] = {
        "Source": [],
        "Tests": [],
        "Docs": [],
        "CI / GitHub": [],
        "Other": [],
    }
    for path in files:
        if path.startswith("src/"):
            buckets["Source"].append(path)
        elif path.startswith("tests/"):
            buckets["Tests"].append(path)
        elif path.startswith("docs/") or path in {
            "README.md",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "SECURITY.md",
            "CODE_OF_CONDUCT.md",
            "AGENTS.md",
        }:
            buckets["Docs"].append(path)
        elif path.startswith(".github/"):
            buckets["CI / GitHub"].append(path)
        else:
            buckets["Other"].append(path)

    lines: list[str] = []
    for name, paths in buckets.items():
        if not paths:
            continue
        shown = paths[:8]
        extra = len(paths) - len(shown)
        joined = ", ".join(f"`{p}`" for p in shown)
        if extra > 0:
            joined += f" (+{extra} more)"
        lines.append(f"- **{name}:** {joined}")
    return lines


def detect_related_issues(subjects: list[str], branch: str) -> list[str]:
    text = " ".join(subjects) + " " + branch
    found: list[str] = []
    for match in re.finditer(
        r"(?:fixes|closes|resolves)\s+#(\d+)|(?:^|[\s(])#(\d+)\b",
        text,
        flags=re.IGNORECASE,
    ):
        num = match.group(1) or match.group(2)
        ref = f"#{num}"
        if ref not in found:
            found.append(ref)
    # branch patterns: fix/123-foo, issue-45
    for match in re.finditer(r"(?:^|/)(?:fix|fixes|issue|issues|pr)[-_/]?(\d+)", branch, re.I):
        ref = f"#{match.group(1)}"
        if ref not in found:
            found.append(ref)
    return found


def build_body(
    *,
    base: str,
    head: str,
    branch: str,
    subjects: list[str],
    files: list[str],
) -> str:
    types = infer_types(subjects, files)
    summary = summarize(subjects, files)
    changes = changes_list(subjects, files)
    issues = detect_related_issues(subjects, branch)
    related = "\n".join(f"- {ref}" for ref in issues) if issues else "- _None detected from commits/branch name._"

    commit_block = ""
    non_merge = [s for s in subjects if not s.lower().startswith("merge ")]
    if non_merge:
        shown = non_merge[:15]
        commit_block = "\n".join(f"- {s}" for s in shown)
        if len(non_merge) > 15:
            commit_block += f"\n- …and {len(non_merge) - 15} more"
    else:
        commit_block = "- _No commits found between base and head._"

    lines = [
        AUTOFILLED_MARKER,
        "",
        "## Summary",
        "",
        summary,
        "",
        "## Motivation / Context",
        "",
        f"Auto-filled from commits on `{branch}` vs `{base}`.",
        "Edit this section with user-facing context or linked discussion if needed.",
        "",
        "## Changes",
        "",
        *[f"- {item}" for item in changes],
        "",
        "### Files touched",
        "",
        *group_files(files),
        "",
        "### Commits",
        "",
        commit_block,
        "",
        "## Type of change",
        "",
        format_type_section(types),
        "",
        "## Testing and verification",
        "",
        "_Auto-fill could not run your tests. Replace with what you actually ran._",
        "",
        "```bash",
        "# python -m pytest tests/<relevant>.py -q",
        "# python -m mana-agent --help",
        "```",
        "",
        "- [ ] Relevant tests pass",
        "- [ ] Manual / CLI smoke check completed (if applicable)",
        "- [ ] No fallback / keyword-routing path introduced (model-decision paths unchanged or improved)",
        "",
        "## Screenshots / CLI output",
        "",
        "_Optional — add terminal output or UI screenshots if useful._",
        "",
        "## Breaking changes",
        "",
        "Yes — describe impact and migration."
        if "breaking" in types
        else "None",
        "",
        "## Related issues",
        "",
        related,
        "",
        "## Author checklist",
        "",
        "- [ ] Change is focused and limited to the requested scope",
        "- [ ] `CHANGELOG.md` updated for user-visible or repository behavior changes",
        "- [ ] Docs / help text updated when user-facing behavior changes",
        "- [ ] Secrets, tokens, and private data are not committed or logged",
        "- [ ] Ready for review",
        "",
        "---",
        "",
        f"_Body auto-filled from `{base}...{head}` "
        f"({len(non_merge)} commit(s), {len(files)} file(s)). "
        "Edit freely; re-open alone will not overwrite a customized body._",
        "",
    ]
    return "\n".join(lines)


def body_should_autofill(existing: str | None) -> bool:
    """Only overwrite empty/template bodies so manual edits are preserved."""
    if existing is None:
        return True
    text = existing.strip()
    if not text:
        return True
    if TEMPLATE_MARKER in text:
        return True
    if AUTOFILLED_MARKER in text:
        # Allow refresh only if still the autofilled draft (optional: skip)
        return False
    # Heuristic: still the static template (placeholder comments / empty bullets)
    template_signals = (
        "<!-- One or two sentences describing what this PR does.",
        "<!-- Why is this change needed?",
        "<!-- Bullet the concrete code",
        "<!-- Check all that apply.",
        "<!-- How was this verified?",
        "<!-- Optional: paste terminal output",
        "<!-- Use closing keywords",
        "<!-- mana-agent:pr-template -->",
    )
    hits = sum(1 for s in template_signals if s in text)
    if hits >= 2:
        return True
    # Mostly headings with empty content
    if text.count("## ") >= 5 and text.count("- [ ]") >= 5 and "Auto-filled from" not in text:
        if "<!--" in text or re.search(r"^-\s*$", text, re.M):
            return True
    return False


def github_api(
    method: str,
    url: str,
    token: str,
    payload: dict | None = None,
) -> dict | list | None:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "mana-agent-pr-autofill",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API {exc.code} {method} {url}: {body}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Base ref/SHA (PR base)")
    parser.add_argument("--head", required=True, help="Head ref/SHA (PR head)")
    parser.add_argument("--branch", default="", help="Head branch name")
    parser.add_argument("--output", default="", help="Write body to file instead of stdout")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="PATCH the pull request body via GitHub API",
    )
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--pr", type=int, default=0, help="Pull request number")
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite even if body no longer looks like the template",
    )
    args = parser.parse_args(argv)

    subjects = commit_subjects(args.base, args.head)
    files = changed_files(args.base, args.head)
    branch = args.branch or run_git("rev-parse", "--abbrev-ref", "HEAD", check=False) or "head"
    body = build_body(
        base=args.base,
        head=args.head,
        branch=branch,
        subjects=subjects,
        files=files,
    )

    if args.output:
        Path(args.output).write_text(body, encoding="utf-8")
        print(f"Wrote {args.output} ({len(body)} bytes)", file=sys.stderr)

    if args.apply:
        if not args.repo or not args.pr or not args.token:
            print("error: --apply requires --repo, --pr, and token", file=sys.stderr)
            return 2
        current = github_api(
            "GET",
            f"https://api.github.com/repos/{args.repo}/pulls/{args.pr}",
            token=args.token,
        )
        assert isinstance(current, dict)
        existing_body = current.get("body") or ""
        if not args.force and not body_should_autofill(existing_body):
            print("PR body already customized; leaving unchanged.", file=sys.stderr)
            return 0
        github_api(
            "PATCH",
            f"https://api.github.com/repos/{args.repo}/pulls/{args.pr}",
            token=args.token,
            payload={"body": body},
        )
        print(f"Updated PR #{args.pr} body ({len(body)} bytes).", file=sys.stderr)
        return 0

    if not args.output:
        sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
