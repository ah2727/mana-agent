from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal, Sequence


VerificationProfile = Literal["documentation_verification", "task_verification", "project_verification"]


@dataclass(frozen=True, slots=True)
class VerificationDecision:
    verification_profile: VerificationProfile
    reason: str
    commands: tuple[str, ...]
    skip_full_pytest_reason: str = ""
    verification_class: str = ""
    skipped_checks: tuple[str, ...] = ()

    def trace_row(self) -> dict[str, object]:
        return {
            "layer": "verification_planner",
            "decision": self.verification_profile,
            "reason": self.reason,
            "commands": list(self.commands),
            "skip_full_pytest_reason": self.skip_full_pytest_reason,
            "verification_class": self.verification_class or self.verification_profile,
            "skipped_checks": list(self.skipped_checks),
        }


@dataclass(frozen=True, slots=True)
class ArtifactVerificationResult:
    ok: bool
    checks: tuple[dict[str, object], ...]
    affected_files: tuple[str, ...]
    failure_code: str = ""
    skipped_checks: tuple[str, ...] = ()

    def trace_row(self) -> dict[str, object]:
        return {
            "tool_name": "verify_changed_artifacts",
            "status": "ok" if self.ok else "error",
            "verification_class": "documentation",
            "affected_files": list(self.affected_files),
            "checks": list(self.checks),
            "planned_commands": ["verify_changed_artifacts"],
            "failure_code": self.failure_code,
            "skipped": False,
            "skipped_checks": list(self.skipped_checks),
            "skipped_commands": [
                {"command": reason.split(":", 1)[0], "reason": reason.split(":", 1)[-1].strip()}
                for reason in self.skipped_checks
            ],
        }


def plan_verification(*, changed_files: Sequence[str], core_agent_change: bool = False) -> VerificationDecision:
    files = [str(path).replace("\\", "/").lstrip("./") for path in changed_files if str(path).strip()]
    docs_only = bool(files) and all(path.lower().endswith((".md", ".txt", ".rst")) for path in files)
    if docs_only and not core_agent_change:
        target = files[0]
        return VerificationDecision(
            verification_profile="documentation_verification",
            reason="Only documentation changed; no source-code behavior changed.",
            commands=("verify_changed_artifacts",),
            skip_full_pytest_reason="README-only documentation change" if target.lower() == "readme.md" else "docs-only documentation change",
            verification_class="documentation",
            skipped_checks=("project pytest", "project build"),
        )
    python_files = [path for path in files if path.lower().endswith(".py")]
    if python_files and not core_agent_change:
        modules = tuple(f"python -m py_compile {path}" for path in python_files)
        return VerificationDecision(
            verification_profile="task_verification",
            reason="A localized Python change requires syntax and targeted-test verification.",
            commands=modules + ("pytest -q <mapped-target-tests>",),
            skip_full_pytest_reason="localized Python change",
            verification_class="code",
            skipped_checks=("full project pytest",),
        )
    return VerificationDecision(
        verification_profile="project_verification",
        reason="Core agent behavior or source files changed.",
        commands=("pytest -q",),
        verification_class="project",
    )


_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def verify_documentation_changes(*, repo_root: Path, changed_files: Sequence[str]) -> ArtifactVerificationResult:
    root = Path(repo_root).resolve()
    checks: list[dict[str, object]] = []
    failures: list[str] = []
    files = tuple(dict.fromkeys(str(path).replace("\\", "/").lstrip("./") for path in changed_files if str(path).strip()))
    for rel in files:
        target = (root / rel).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            failures.append(f"path_outside_repository:{rel}")
            continue
        try:
            content = target.read_text(encoding="utf-8")
        except OSError as exc:
            failures.append(f"read_failed:{rel}:{exc}")
            continue
        headings = [" ".join(line.lstrip("#").strip().split()).casefold() for line in content.splitlines() if line.startswith("#")]
        duplicates = sorted({heading for heading in headings if heading and headings.count(heading) > 1})
        checks.append({"name": "duplicate_headings", "path": rel, "status": "passed" if not duplicates else "failed", "duplicates": duplicates})
        if duplicates:
            failures.append(f"duplicate_headings:{rel}")
        broken_links: list[str] = []
        for raw_link in _MARKDOWN_LINK_RE.findall(content):
            link = raw_link.strip().split("#", 1)[0].strip()
            if not link or "://" in link or link.startswith(("mailto:", "#")):
                continue
            linked = (target.parent / link).resolve()
            try:
                linked.relative_to(root)
            except ValueError:
                broken_links.append(raw_link)
                continue
            if not linked.exists():
                broken_links.append(raw_link)
        checks.append({"name": "local_links", "path": rel, "status": "passed" if not broken_links else "failed", "broken_links": broken_links})
        if broken_links:
            failures.append(f"broken_local_links:{rel}")
        checks.append({"name": "content_reread", "path": rel, "status": "passed" if content.strip() else "failed", "bytes": len(content.encode("utf-8"))})
        if not content.strip():
            failures.append(f"empty_document:{rel}")
    return ArtifactVerificationResult(
        ok=not failures,
        checks=tuple(checks),
        affected_files=files,
        failure_code="" if not failures else "documentation_verification_failed",
        skipped_checks=("project pytest: documentation-only change", "project build: documentation-only change"),
    )


__all__ = [
    "ArtifactVerificationResult",
    "VerificationDecision",
    "VerificationProfile",
    "plan_verification",
    "verify_documentation_changes",
]
