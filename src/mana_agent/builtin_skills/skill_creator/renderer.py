from __future__ import annotations

import json
from typing import Any

from .schema import ProposalEvidence, ProposalManifest, SkillDraft, ValidationReport


def _yaml_text(value: dict[str, Any]) -> str:
    # JSON is a strict subset of YAML 1.2 and avoids adding a second serializer.
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def render_proposal_yaml(manifest: ProposalManifest) -> str:
    return _yaml_text(manifest.model_dump(mode="json"))


def render_skill_markdown(draft: SkillDraft, *, evidence: ProposalEvidence) -> str:
    def bullets(values: list[str]) -> str:
        return "\n".join(f"- {item}" for item in values)

    frontmatter = [
        "---",
        f"name: {draft.name}",
        f"description: {json.dumps(draft.description, ensure_ascii=False)}",
        "triggers:",
        *[f"  - {json.dumps(item, ensure_ascii=False)}" for item in draft.triggers],
        "required_tools:",
        *[f"  - {item}" for item in draft.required_tools],
        "required_permissions:",
        *[f"  - {item}" for item in draft.required_permissions],
        f"risk_level: {draft.risk_level}",
        "version: 1.0.0",
        "---",
    ]
    return "\n".join(
        [
            *frontmatter,
            "",
            "# Purpose",
            "",
            draft.purpose,
            "",
            "# When to use",
            "",
            bullets(draft.when_to_use),
            "",
            "# When not to use",
            "",
            bullets(draft.when_not_to_use),
            "",
            "# Preconditions",
            "",
            bullets(draft.preconditions),
            "",
            "# Procedure",
            "",
            "\n".join(f"{index}. {item}" for index, item in enumerate(draft.procedure, start=1)),
            "",
            "# Safety constraints",
            "",
            bullets(draft.safety_constraints),
            "",
            "# Verification",
            "",
            bullets(draft.verification),
            "",
            "# Failure recovery",
            "",
            bullets(draft.failure_recovery),
            "",
            "# Evidence provenance",
            "",
            f"Derived from recorded sessions: {', '.join(evidence.session_ids)}.",
            f"Recorded tasks: {', '.join(evidence.task_ids)}. Full redacted evidence remains with the proposal.",
            "",
        ]
    )


def render_readme(manifest: ProposalManifest, report: ValidationReport) -> str:
    warnings = [item for item in report.findings if item.severity in {"warning", "error", "critical"}]
    warning_text = "\n".join(f"- [{item.severity}] {item.message}" for item in warnings) or "- None."
    return f"""# {manifest.display_name}

This is an Experience-to-Skill proposal. It is not active and cannot be loaded by agents until a user explicitly installs it.

- Proposal: `{manifest.proposal_id}`
- Status: `{manifest.status}`
- Confidence: `{manifest.confidence:.2f}`
- Risk: `{manifest.risk.get('level', 'unknown')}`
- Required tools: {', '.join(manifest.required_tools) or 'none'}
- Required permissions: {', '.join(manifest.required_permissions) or 'none'}
- Duplicate: {manifest.duplicate_of or 'none detected'}

## Validation warnings

{warning_text}

Review `SKILL.md`, `evidence.json`, and `validation.json` before installing.
"""
