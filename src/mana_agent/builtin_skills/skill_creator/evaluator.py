from __future__ import annotations

from .schema import EligibilityDecision, ExperienceRecord, WorkshopConfig


class ExperienceEvaluator:
    """Deterministic minimum gates and evidence-based confidence scoring."""

    DISABLED_COMPONENTS = frozenset({"skill_creator", "proposal_validator", "proposal_installer"})

    def __init__(self, config: WorkshopConfig | None = None) -> None:
        self.config = config or WorkshopConfig.load()

    def evaluate(self, experience: ExperienceRecord) -> EligibilityDecision:
        reasons: list[str] = []
        signals: list[str] = []
        penalties: list[str] = []

        if not self.config.enabled:
            reasons.append("Experience-to-Skill Workshop is disabled.")
        if experience.source_component in self.DISABLED_COMPONENTS:
            reasons.append("Recursive experience learning is disabled for this component.")
        if experience.status != "completed":
            reasons.append("Only completed tasks are eligible.")
        if len(experience.workflow_steps) < 2:
            reasons.append("A reusable multi-step workflow was not recorded.")
        if not experience.changed_files and not experience.artifact_references:
            reasons.append("No meaningful mutation or artifact was recorded.")
        if self.config.require_verification and not experience.verification_passed:
            reasons.append("Successful verification is required.")
        if experience.unresolved_warnings:
            reasons.append("Unresolved verification warnings remain.")
        if experience.repository_specificity == "high":
            reasons.append("The workflow is too repository-specific to generalize safely.")
        if experience.successful_runs < self.config.minimum_successful_runs:
            reasons.append("The minimum successful-run count was not met.")
        if self.config.require_user_acceptance and not experience.user_accepted:
            reasons.append("Explicit user acceptance is required.")

        score = 0.0
        if experience.verification_passed:
            score += 0.25
            signals.append("successful_verification:+0.25")
        if experience.user_accepted:
            score += 0.20
            signals.append("user_accepted:+0.20")
        if experience.successful_runs > 1:
            score += 0.15
            signals.append("multiple_successful_runs:+0.15")
        if experience.reusable_trigger_present:
            score += 0.10
            signals.append("clear_reusable_trigger:+0.10")
        if experience.deterministic_verification:
            score += 0.10
            signals.append("deterministic_verification:+0.10")
        if experience.repository_specificity == "low":
            score += 0.10
            signals.append("low_repository_specificity:+0.10")
        if not experience.unresolved_warnings:
            score += 0.10
            signals.append("no_unresolved_warnings:+0.10")

        if experience.unresolved_corrections:
            score -= 0.30
            penalties.append("unresolved_user_correction:-0.30")
        if experience.verification_results and not experience.verification_passed:
            score -= 0.25
            penalties.append("verification_failed_or_partial:-0.25")
        if experience.repository_specificity == "high":
            score -= 0.20
            penalties.append("high_repository_specificity:-0.20")
        if not experience.safety_constraints_present:
            score -= 0.25
            penalties.append("missing_safety_constraints:-0.25")
        if not experience.verification_results or not experience.decisions:
            score -= 0.15
            penalties.append("incomplete_evidence:-0.15")

        confidence = round(max(0.0, min(1.0, score)), 4)
        if reasons:
            status = "quarantine" if not experience.safety_constraints_present else "ineligible"
            return EligibilityDecision(
                eligible=False,
                confidence=confidence,
                status=status,
                signals=signals,
                penalties=penalties,
                reasons=reasons,
            )
        if confidence < self.config.needs_attention_confidence:
            return EligibilityDecision(
                eligible=False,
                confidence=confidence,
                status="ineligible",
                signals=signals,
                penalties=penalties,
                reasons=["Confidence is below the configured proposal threshold."],
            )
        status = "pending_review" if confidence >= self.config.minimum_confidence else "needs_attention"
        return EligibilityDecision(
            eligible=True,
            confidence=confidence,
            status=status,
            signals=signals,
            penalties=penalties,
        )
