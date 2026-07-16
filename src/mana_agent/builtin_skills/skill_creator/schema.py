from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mana_agent.config.user_config import get_setting
from mana_agent.workspaces.paths import mana_home

SCHEMA_VERSION = 1
SUPPORTED_PERMISSIONS = frozenset(
    {
        "repository_read",
        "repository_write",
        "command_execution",
        "network_access",
        "git_read",
        "git_write",
    }
)
PROPOSAL_STATUSES = (
    "pending_review",
    "needs_attention",
    "installed",
    "rejected",
    "quarantined",
)
RISK_LEVELS = ("low", "medium", "high", "critical")
_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{1,79}$")
_PROPOSAL_ID = re.compile(r"^skill_proposal_[0-9]{8}_[0-9]{4}_[a-z0-9][a-z0-9-]{1,79}$")


class WorkshopConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    auto_propose: bool = True
    minimum_confidence: float = Field(default=0.80, ge=0, le=1)
    needs_attention_confidence: float = Field(default=0.60, ge=0, le=1)
    minimum_successful_runs: int = Field(default=1, ge=1)
    require_verification: bool = True
    require_user_acceptance: bool = False
    semantic_duplicate_threshold: float = Field(default=0.88, ge=0, le=1)
    retain_rejected_days: int = Field(default=90, ge=0)
    quarantine_on_validation_failure: bool = True
    skills_path: Path | None = None
    proposals_path: Path | None = None
    quarantine_path: Path | None = None

    @model_validator(mode="after")
    def validate_thresholds(self) -> "WorkshopConfig":
        if self.needs_attention_confidence > self.minimum_confidence:
            raise ValueError("needs_attention_confidence cannot exceed minimum_confidence")
        return self

    @classmethod
    def load(cls) -> "WorkshopConfig":
        table = get_setting("experience_to_skill", {})
        values = dict(table) if isinstance(table, dict) else {}
        env_map = {
            "enabled": "MANA_EXPERIENCE_TO_SKILL_ENABLED",
            "auto_propose": "MANA_EXPERIENCE_TO_SKILL_AUTO_PROPOSE",
            "minimum_confidence": "MANA_EXPERIENCE_TO_SKILL_MINIMUM_CONFIDENCE",
            "needs_attention_confidence": "MANA_EXPERIENCE_TO_SKILL_NEEDS_ATTENTION_CONFIDENCE",
            "minimum_successful_runs": "MANA_EXPERIENCE_TO_SKILL_MINIMUM_SUCCESSFUL_RUNS",
            "require_verification": "MANA_EXPERIENCE_TO_SKILL_REQUIRE_VERIFICATION",
            "require_user_acceptance": "MANA_EXPERIENCE_TO_SKILL_REQUIRE_USER_ACCEPTANCE",
            "semantic_duplicate_threshold": "MANA_EXPERIENCE_TO_SKILL_DUPLICATE_THRESHOLD",
            "retain_rejected_days": "MANA_EXPERIENCE_TO_SKILL_RETAIN_REJECTED_DAYS",
            "quarantine_on_validation_failure": "MANA_EXPERIENCE_TO_SKILL_QUARANTINE_ON_FAILURE",
            "skills_path": "MANA_SKILLS_ROOT",
            "proposals_path": "MANA_SKILL_PROPOSALS_ROOT",
            "quarantine_path": "MANA_SKILL_QUARANTINE_ROOT",
        }
        bool_fields = {
            "enabled", "auto_propose", "require_verification", "require_user_acceptance",
            "quarantine_on_validation_failure",
        }
        for field, env_name in env_map.items():
            raw = os.getenv(env_name)
            if raw is None or not raw.strip():
                continue
            if field in bool_fields:
                values[field] = raw.strip().lower() in {"1", "true", "yes", "on"}
            else:
                values[field] = raw.strip()
        return cls.model_validate(values)


class ExperienceRecord(BaseModel):
    """Recorded task facts. Model-generated evidence is not accepted here."""

    schema_version: int = SCHEMA_VERSION
    session_id: str
    task_id: str
    summary: str
    result: str = ""
    status: Literal["completed", "failed", "abandoned"] = "completed"
    workflow_steps: list[str] = Field(default_factory=list)
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    artifact_references: list[str] = Field(default_factory=list)
    verification_commands: list[str] = Field(default_factory=list)
    verification_results: list[dict[str, Any]] = Field(default_factory=list)
    verification_passed: bool = False
    successful_runs: int = 1
    failed_runs: int = 0
    user_corrections: list[str] = Field(default_factory=list)
    unresolved_corrections: bool = False
    user_accepted: bool = False
    reusable_trigger_present: bool = False
    deterministic_verification: bool = False
    repository_specificity: Literal["low", "medium", "high"] = "medium"
    safety_constraints_present: bool = True
    unresolved_warnings: list[str] = Field(default_factory=list)
    agent_ids: list[str] = Field(default_factory=list)
    subagent_ids: list[str] = Field(default_factory=list)
    failure_recovery: list[dict[str, Any]] = Field(default_factory=list)
    git: dict[str, Any] = Field(default_factory=dict)
    tool_result_hashes: dict[str, str] = Field(default_factory=dict)
    source_component: str = "task_completion"


class EligibilityDecision(BaseModel):
    eligible: bool
    confidence: float = Field(ge=0, le=1)
    status: Literal["pending_review", "needs_attention", "ineligible", "quarantine"]
    signals: list[str] = Field(default_factory=list)
    penalties: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class SkillDraft(BaseModel):
    """Structured output required from the trusted model generator."""

    name: str
    display_name: str
    description: str
    triggers: list[str] = Field(min_length=1)
    required_tools: list[str] = Field(default_factory=list)
    required_permissions: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high", "critical"] = "medium"
    risk_reasons: list[str] = Field(default_factory=list)
    purpose: str
    when_to_use: list[str] = Field(min_length=1)
    when_not_to_use: list[str] = Field(min_length=1)
    preconditions: list[str] = Field(min_length=1)
    procedure: list[str] = Field(min_length=2)
    safety_constraints: list[str] = Field(min_length=1)
    verification: list[str] = Field(min_length=1)
    failure_recovery: list[str] = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _SLUG.fullmatch(normalized) or ".." in normalized:
            raise ValueError("skill name must be a safe lowercase slug")
        if normalized == "skill-creator":
            raise ValueError("the trusted skill-creator capability cannot be replaced")
        return normalized

    @field_validator("required_permissions")
    @classmethod
    def validate_permissions(cls, value: list[str]) -> list[str]:
        unsupported = sorted(set(value) - SUPPORTED_PERMISSIONS)
        if unsupported:
            raise ValueError("unsupported permissions: " + ", ".join(unsupported))
        return list(dict.fromkeys(value))


class ProposalManifest(BaseModel):
    schema_version: int = SCHEMA_VERSION
    proposal_id: str
    name: str
    display_name: str
    description: str
    status: Literal["pending_review", "needs_attention", "installed", "rejected", "quarantined"] = "pending_review"
    created_at: str
    updated_at: str
    source_sessions: list[str]
    source_tasks: list[str]
    confidence: float = Field(ge=0, le=1)
    successful_runs: int = 1
    failed_runs: int = 0
    triggers: list[str]
    required_tools: list[str]
    required_permissions: list[str]
    verification: dict[str, list[str]]
    risk: dict[str, Any]
    evidence_summary: dict[str, Any]
    duplicate_of: str | None = None
    quarantine_reason: str | None = None
    aliases: list[str] = Field(default_factory=list)
    version: str = "1.0.0"
    installed_at: str | None = None
    rejection_reason: str | None = None

    @field_validator("proposal_id")
    @classmethod
    def validate_proposal_id(cls, value: str) -> str:
        if not _PROPOSAL_ID.fullmatch(value):
            raise ValueError("invalid proposal identifier")
        return value

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not _SLUG.fullmatch(value) or ".." in value:
            raise ValueError("invalid skill name")
        return value

    @field_validator("required_permissions")
    @classmethod
    def validate_permissions(cls, value: list[str]) -> list[str]:
        unsupported = sorted(set(value) - SUPPORTED_PERMISSIONS)
        if unsupported:
            raise ValueError("unsupported permissions: " + ", ".join(unsupported))
        return value


class ProposalEvidence(BaseModel):
    schema_version: int = SCHEMA_VERSION
    session_ids: list[str]
    task_ids: list[str]
    agent_ids: list[str] = Field(default_factory=list)
    subagent_ids: list[str] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    artifact_references: list[str] = Field(default_factory=list)
    verification_commands: list[str] = Field(default_factory=list)
    verification_results: list[dict[str, Any]] = Field(default_factory=list)
    user_corrections: list[str] = Field(default_factory=list)
    user_accepted: bool = False
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    failure_recovery: list[dict[str, Any]] = Field(default_factory=list)
    git: dict[str, Any] = Field(default_factory=dict)
    tool_result_hashes: dict[str, str] = Field(default_factory=dict)
    redactions: list[str] = Field(default_factory=list)


class ValidationFinding(BaseModel):
    severity: Literal["info", "warning", "error", "critical"]
    code: str
    message: str


class ValidationReport(BaseModel):
    schema_version: int = SCHEMA_VERSION
    valid: bool
    checked_at: str
    findings: list[ValidationFinding] = Field(default_factory=list)
    duplicate_analysis: dict[str, Any] = Field(default_factory=dict)
    content_sha256: str = ""


def default_workshop_paths(config: WorkshopConfig | None = None) -> tuple[Path, Path, Path]:
    config = config or WorkshopConfig.load()
    home = mana_home()
    return (
        Path(config.skills_path or home / "skills").expanduser().resolve(),
        Path(config.proposals_path or home / "skill-proposals").expanduser().resolve(),
        Path(config.quarantine_path or home / "skill-quarantine").expanduser().resolve(),
    )
