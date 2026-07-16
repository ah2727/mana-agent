from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

from pydantic import BaseModel

from .evaluator import ExperienceEvaluator
from .renderer import render_skill_markdown
from .schema import (
    EligibilityDecision,
    ExperienceRecord,
    ProposalEvidence,
    ProposalManifest,
    SkillDraft,
    ValidationReport,
    WorkshopConfig,
)
from .storage import ProposalStorage
from .validator import ProposalValidator, redact_value

logger = logging.getLogger(__name__)


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class DraftGenerator(Protocol):
    """Model boundary: implementations must return a validated structured draft."""

    def generate(self, experience: ExperienceRecord, decision: EligibilityDecision) -> SkillDraft: ...


@dataclass(frozen=True, slots=True)
class SkillProposalResult:
    decision: EligibilityDecision
    proposal: ProposalManifest | None = None
    path: Path | None = None
    validation: ValidationReport | None = None
    merged_into: str | None = None


class SkillCreator:
    """Trusted proposal orchestrator. It never installs or activates skills."""

    def __init__(
        self,
        *,
        config: WorkshopConfig | None = None,
        storage: ProposalStorage | None = None,
        evaluator: ExperienceEvaluator | None = None,
        validator: ProposalValidator | None = None,
        event_sink: Callable[[str, dict[str, object]], None] | None = None,
    ) -> None:
        self.config = config or WorkshopConfig.load()
        self.storage = storage or ProposalStorage(config=self.config)
        self.evaluator = evaluator or ExperienceEvaluator(self.config)
        self.validator = validator or ProposalValidator()
        self.event_sink = event_sink or _default_event_sink

    def create(self, experience: ExperienceRecord, generator: DraftGenerator) -> SkillProposalResult:
        decision = self.evaluator.evaluate(experience)
        if not decision.eligible:
            return SkillProposalResult(decision)
        self._emit("skill_candidate_detected", task_id=experience.task_id, confidence=decision.confidence)
        self._emit("skill_proposal_generation_started", task_id=experience.task_id)
        # The model sees the same recursively redacted record that can later be
        # persisted; private connector/tool payloads never enter its prompt.
        safe_experience = ExperienceRecord.model_validate(redact_value(experience.model_dump(mode="json")))
        draft = generator.generate(safe_experience, decision)
        return self.create_from_draft(safe_experience, draft, decision=decision)

    def create_from_draft(
        self,
        experience: ExperienceRecord,
        draft: SkillDraft,
        *,
        decision: EligibilityDecision | None = None,
    ) -> SkillProposalResult:
        decision = decision or self.evaluator.evaluate(experience)
        if not decision.eligible:
            return SkillProposalResult(decision)
        redactions: list[str] = []
        redacted = redact_value(experience.model_dump(mode="json"), redactions=redactions)
        safe_experience = ExperienceRecord.model_validate(redacted)
        evidence = ProposalEvidence(
            session_ids=[safe_experience.session_id],
            task_ids=[safe_experience.task_id],
            agent_ids=safe_experience.agent_ids,
            subagent_ids=safe_experience.subagent_ids,
            changed_files=safe_experience.changed_files,
            artifact_references=safe_experience.artifact_references,
            verification_commands=safe_experience.verification_commands,
            verification_results=safe_experience.verification_results,
            user_corrections=safe_experience.user_corrections,
            user_accepted=safe_experience.user_accepted,
            decisions=safe_experience.decisions,
            failure_recovery=safe_experience.failure_recovery,
            git=safe_experience.git,
            tool_result_hashes=safe_experience.tool_result_hashes,
            redactions=list(dict.fromkeys(redactions)),
        )
        duplicate = self._find_duplicate(draft)
        if duplicate:
            merged = self._merge_duplicate(duplicate, safe_experience, evidence)
            self._emit("skill_duplicate_detected", existing_skill=duplicate)
            return SkillProposalResult(decision, merged_into=merged)

        now = utcnow()
        proposal_id = self._proposal_id(safe_experience, draft.name)
        manifest = ProposalManifest(
            proposal_id=proposal_id,
            name=draft.name,
            display_name=draft.display_name,
            description=draft.description,
            status=decision.status,
            created_at=now,
            updated_at=now,
            source_sessions=[safe_experience.session_id],
            source_tasks=[safe_experience.task_id],
            confidence=decision.confidence,
            successful_runs=safe_experience.successful_runs,
            failed_runs=safe_experience.failed_runs,
            triggers=draft.triggers,
            required_tools=draft.required_tools,
            required_permissions=draft.required_permissions,
            verification={"commands": safe_experience.verification_commands, "expected": draft.verification},
            risk={"level": draft.risk_level, "reasons": draft.risk_reasons},
            evidence_summary={
                "files_changed": len(safe_experience.changed_files),
                "tests_passed": sum(1 for row in safe_experience.verification_results if bool(row.get("passed", row.get("success", False)))),
                "user_accepted": safe_experience.user_accepted,
                "eligibility_signals": decision.signals,
                "confidence_penalties": decision.penalties,
            },
            aliases=draft.aliases,
        )
        markdown = render_skill_markdown(draft, evidence=evidence)
        report = self.validator.validate(manifest, markdown, evidence)
        quarantine = not report.valid and self.config.quarantine_on_validation_failure
        if quarantine:
            manifest.status = "quarantined"
            manifest.quarantine_reason = "; ".join(item.message for item in report.findings if item.severity in {"error", "critical"})
            self._emit("skill_proposal_validation_failed", proposal_id=proposal_id, reason=manifest.quarantine_reason)
        elif not report.valid:
            self._emit("skill_proposal_validation_failed", proposal_id=proposal_id)
            return SkillProposalResult(decision, proposal=manifest, validation=report)
        try:
            path = self.storage.write(manifest, markdown, evidence, report, quarantine=quarantine)
        except FileExistsError:
            # Another process won the same stable proposal-id race. Merge the
            # newly recorded run instead of creating a second proposal.
            merged = self._merge_duplicate(proposal_id, safe_experience, evidence)
            self._emit("skill_duplicate_detected", existing_skill=proposal_id)
            return SkillProposalResult(decision, merged_into=merged)
        self._emit("skill_proposal_quarantined" if quarantine else "skill_proposal_created", proposal_id=proposal_id, name=draft.name)
        logger.info("[skill-workshop] proposal_%s proposal_id=%s name=%s", "quarantined" if quarantine else "created", proposal_id, draft.name)
        return SkillProposalResult(decision, proposal=manifest, path=path, validation=report)

    def edit(self, proposal_id: str, *, draft: SkillDraft | None = None, markdown: str | None = None) -> ProposalManifest:
        path, manifest, evidence, _report, current_markdown = self.storage.load(proposal_id)
        if manifest.status in {"installed", "quarantined"}:
            raise ValueError(f"proposal cannot be edited in status: {manifest.status}")
        if draft is not None:
            manifest.name = draft.name
            manifest.display_name = draft.display_name
            manifest.description = draft.description
            manifest.triggers = draft.triggers
            manifest.required_tools = draft.required_tools
            manifest.required_permissions = draft.required_permissions
            manifest.risk = {"level": draft.risk_level, "reasons": draft.risk_reasons}
            manifest.aliases = draft.aliases
            current_markdown = render_skill_markdown(draft, evidence=evidence)
        elif markdown is not None:
            current_markdown = markdown
        else:
            raise ValueError("edit requires a structured draft or SKILL.md content")
        manifest.status = "needs_attention"
        manifest.updated_at = utcnow()
        report = self.validator.validate(manifest, current_markdown, evidence)
        self.storage.update(path, manifest, current_markdown, evidence, report)
        return manifest

    def _find_duplicate(self, draft: SkillDraft) -> str | None:
        candidate = " ".join([draft.name, draft.description, *draft.triggers, *draft.procedure])
        best: tuple[float, str] = (0.0, "")
        for manifest in self.storage.list():
            existing = " ".join([manifest.name, manifest.description, *manifest.triggers, *manifest.aliases])
            score = _cosine_similarity(candidate, existing)
            if draft.name == manifest.name or draft.name in manifest.aliases:
                score = 1.0
            if score > best[0]:
                best = (score, manifest.proposal_id)
        for path in self.storage.paths.skills.glob("*/SKILL.md"):
            content = path.read_text(encoding="utf-8", errors="replace")[:12000]
            score = 1.0 if path.parent.name == draft.name else _cosine_similarity(candidate, content)
            if score > best[0]:
                best = (score, f"active:{path.parent.name}")
        return best[1] if best[0] >= self.config.semantic_duplicate_threshold else None

    def _merge_duplicate(self, duplicate: str, experience: ExperienceRecord, evidence: ProposalEvidence) -> str:
        if duplicate.startswith("active:"):
            # Active skills are immutable through proposal generation. Store only
            # append-only supporting evidence next to workshop state.
            name = duplicate.split(":", 1)[1]
            target = self.storage.paths.proposals / "duplicate-evidence" / name
            target.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(f"{experience.session_id}:{experience.task_id}".encode()).hexdigest()[:16]
            self.storage._write_json(target / f"{digest}.json", evidence.model_dump(mode="json"))
            return duplicate
        path, manifest, old_evidence, report, markdown = self.storage.load(duplicate)
        if experience.session_id not in manifest.source_sessions:
            manifest.source_sessions.append(experience.session_id)
        if experience.task_id not in manifest.source_tasks:
            manifest.source_tasks.append(experience.task_id)
        manifest.successful_runs += experience.successful_runs
        manifest.failed_runs += experience.failed_runs
        manifest.updated_at = utcnow()
        manifest.confidence = round(max(manifest.confidence, self.evaluator.evaluate(experience).confidence), 4)
        old_evidence.session_ids = list(dict.fromkeys([*old_evidence.session_ids, *evidence.session_ids]))
        old_evidence.task_ids = list(dict.fromkeys([*old_evidence.task_ids, *evidence.task_ids]))
        old_evidence.changed_files = list(dict.fromkeys([*old_evidence.changed_files, *evidence.changed_files]))
        old_evidence.verification_results.extend(evidence.verification_results)
        self.storage.update(path, manifest, markdown, old_evidence, report)
        return duplicate

    @staticmethod
    def _proposal_id(experience: ExperienceRecord, name: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        material = f"{experience.session_id}:{experience.task_id}:{name}"
        sequence = int(hashlib.sha256(material.encode()).hexdigest()[:8], 16) % 10000
        return f"skill_proposal_{timestamp}_{sequence:04d}_{name}"

    def _emit(self, event_type: str, **metadata: object) -> None:
        if self.event_sink is not None:
            self.event_sink(event_type, metadata)


def _cosine_similarity(left: str, right: str) -> float:
    def counts(text: str) -> dict[str, int]:
        values: dict[str, int] = {}
        for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", text.lower()):
            values[token] = values.get(token, 0) + 1
        return values
    a, b = counts(left), counts(right)
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(key, 0) for key, value in a.items())
    return dot / math.sqrt(sum(value * value for value in a.values()) * sum(value * value for value in b.values()))


def _default_event_sink(event_type: str, metadata: dict[str, object]) -> None:
    try:
        from mana_agent.services.execution_event_hub import get_execution_event_hub
        get_execution_event_hub().emit(
            event_type,
            title=event_type.replace("_", " ").title(),
            conversation_id=str(metadata.get("session_id") or ""),
            execution_id=str(metadata.get("task_id") or ""),
            status="failed" if event_type == "skill_proposal_validation_failed" else "success",
            metadata=metadata,
        )
    except Exception:
        pass
