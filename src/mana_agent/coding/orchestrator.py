"""Execute an already validated model backend decision."""

from __future__ import annotations

from mana_agent.coding.models import CodingBackendDecision, CodingTask, CodingTaskResult, WorkspaceContext
from mana_agent.coding.registry import CodingBackendDecisionError, CodingBackendRegistry


class CodingBackendOrchestrator:
    def __init__(self, registry: CodingBackendRegistry) -> None:
        self.registry = registry

    async def execute(
        self,
        decision: CodingBackendDecision,
        task: CodingTask,
        workspace: WorkspaceContext,
    ) -> CodingTaskResult:
        if decision.requires_repository_write != task.requires_repository_write:
            raise CodingBackendDecisionError(
                "Model decision and coding task disagree about repository write access. No backend was executed."
            )
        if task.requires_repository_write and workspace.sandbox != "workspaceWrite":
            raise CodingBackendDecisionError(
                "Writing task received a read-only workspace. No backend was executed."
            )
        backend = self.registry.resolve(decision)
        await backend.start()
        return await backend.execute(task, workspace)


__all__ = ["CodingBackendOrchestrator"]
