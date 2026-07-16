from __future__ import annotations

import logging
from dataclasses import dataclass

from .generator import DraftGenerator, SkillCreator, SkillProposalResult
from .schema import ExperienceRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WorkshopHookResult:
    original_task_succeeded: bool
    proposal_result: SkillProposalResult | None = None
    warning: str | None = None


class ExperienceWorkshopHook:
    """Non-recursive completion hook whose failures never change task status."""

    def __init__(self, creator: SkillCreator | None = None) -> None:
        self.creator = creator or SkillCreator()

    def run(
        self,
        experience: ExperienceRecord,
        *,
        generator: DraftGenerator | None,
        original_task_succeeded: bool,
    ) -> WorkshopHookResult:
        if not original_task_succeeded:
            return WorkshopHookResult(False)
        if not self.creator.config.auto_propose:
            return WorkshopHookResult(True)
        try:
            decision = self.creator.evaluator.evaluate(experience)
            if not decision.eligible:
                return WorkshopHookResult(True, SkillProposalResult(decision))
            if generator is None:
                warning = "Model decision failed: skill proposal generation. No fallback action was executed."
                logger.warning("[skill-workshop] proposal_generation_skipped task_id=%s reason=model_unavailable", experience.task_id)
                self.creator._emit("skill_proposal_validation_failed", task_id=experience.task_id, reason="model_unavailable")
                return WorkshopHookResult(True, warning=warning)
            return WorkshopHookResult(True, self.creator.create(experience, generator))
        except Exception as exc:  # the original completed task remains successful by contract
            warning = f"Skill proposal generation failed: {exc}"
            logger.warning("[skill-workshop] proposal_generation_failed task_id=%s error=%s", experience.task_id, exc)
            self.creator._emit("skill_proposal_validation_failed", task_id=experience.task_id, reason=str(exc))
            return WorkshopHookResult(True, warning=warning)
