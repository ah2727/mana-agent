from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .schema import (
    ProposalEvidence,
    ProposalManifest,
    SUPPORTED_PERMISSIONS,
    ValidationFinding,
    ValidationReport,
)

_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|password|secret)\b\s*[:=]\s*['\"]?([^\s'\"]{6,})"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:sk|ghp|github_pat)_[A-Za-z0-9_-]{12,}\b"),
)
_ABSOLUTE_PERSONAL_PATH = re.compile(r"(?m)(?:^|\s)/(?:Users|home)/[^\s`]+")
_DANGEROUS = ("rm -rf", "git reset --hard", "git push --force", "curl | sh", "sudo ")
_APPROVAL_RECORD_MUTATION = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(?:edit|modify|write|delete|overwrite|patch)\b.{0,80}\b(?:proposal\.yaml|validation\.json|evidence\.json|approval record)\b"
)
_REQUIRED_SECTIONS = (
    "# Purpose",
    "# When to use",
    "# When not to use",
    "# Preconditions",
    "# Procedure",
    "# Safety constraints",
    "# Verification",
    "# Failure recovery",
    "# Evidence provenance",
)


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def redact_value(value: object, *, redactions: list[str] | None = None) -> object:
    """Recursively redact sensitive values before proposal persistence."""
    notes = redactions if redactions is not None else []
    if isinstance(value, dict):
        output: dict[str, object] = {}
        for key, item in value.items():
            if re.search(r"(?i)(api[_-]?key|token|password|secret|private[_-]?key)", str(key)):
                output[str(key)] = "[REDACTED]"
                notes.append(f"redacted sensitive field: {key}")
            else:
                output[str(key)] = redact_value(item, redactions=notes)
        return output
    if isinstance(value, list):
        return [redact_value(item, redactions=notes) for item in value]
    if isinstance(value, str):
        text = value
        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                text = pattern.sub("[REDACTED]", text)
                notes.append("redacted sensitive text")
        return text
    return value


class ProposalValidator:
    def validate(
        self,
        manifest: ProposalManifest,
        markdown: str,
        evidence: ProposalEvidence,
        *,
        duplicate_analysis: dict[str, object] | None = None,
    ) -> ValidationReport:
        findings: list[ValidationFinding] = []
        for section in _REQUIRED_SECTIONS:
            if section not in markdown:
                findings.append(ValidationFinding(severity="error", code="missing_section", message=f"Missing {section}."))
        unsupported = sorted(set(manifest.required_permissions) - SUPPORTED_PERMISSIONS)
        if unsupported:
            findings.append(ValidationFinding(severity="critical", code="unsupported_permission", message="Unsupported permissions: " + ", ".join(unsupported)))
        if not evidence.verification_results or not manifest.verification.get("commands"):
            findings.append(ValidationFinding(severity="error", code="missing_verification", message="Recorded verification evidence and commands are required."))
        combined = markdown + "\n" + json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False)
        for pattern in _SECRET_PATTERNS:
            if pattern.search(combined):
                findings.append(ValidationFinding(severity="critical", code="secret", message="Possible secret material detected."))
                break
        if _ABSOLUTE_PERSONAL_PATH.search(combined):
            findings.append(ValidationFinding(severity="error", code="personal_path", message="Absolute personal paths are not allowed."))
        if ".." in manifest.name or "/" in manifest.name or "\\" in manifest.name:
            findings.append(ValidationFinding(severity="critical", code="path_traversal", message="Unsafe proposal name."))
        lowered = combined.lower()
        for command in _DANGEROUS:
            if command in lowered and "explicit user approval" not in lowered:
                findings.append(ValidationFinding(severity="critical", code="unsafe_procedure", message=f"Destructive procedure lacks explicit approval constraint: {command}"))
        if _APPROVAL_RECORD_MUTATION.search(combined):
            findings.append(ValidationFinding(severity="critical", code="self_approval_mutation", message="A generated skill cannot modify proposal evidence, validation, or approval records."))
        frontmatter_name = re.search(r"(?m)^name:\s*([a-z0-9-]+)\s*$", markdown)
        if frontmatter_name and frontmatter_name.group(1) != manifest.name:
            findings.append(ValidationFinding(severity="error", code="name_mismatch", message="SKILL.md name does not match proposal metadata."))
        if "edit line " in lowered or re.search(r"\bline\s+\d+\b", lowered):
            findings.append(ValidationFinding(severity="warning", code="over_specific", message="Procedure may replay repository-specific line edits."))
        duplicate = dict(duplicate_analysis or {})
        if duplicate.get("match"):
            findings.append(ValidationFinding(severity="warning", code="duplicate", message=f"Likely duplicate of {duplicate.get('match')}."))
        valid = not any(item.severity in {"error", "critical"} for item in findings)
        return ValidationReport(
            valid=valid,
            checked_at=utcnow(),
            findings=findings,
            duplicate_analysis=duplicate,
            content_sha256=hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
        )


def validate_managed_path(path: Path, roots: Iterable[Path]) -> Path:
    resolved = path.expanduser().resolve()
    if not any(_is_relative_to(resolved, root.expanduser().resolve()) for root in roots):
        raise ValueError(f"path is outside configured skill directories: {resolved}")
    return resolved


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
