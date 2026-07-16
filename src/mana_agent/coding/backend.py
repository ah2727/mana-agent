"""Backend protocol implemented by native and external coding executors."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from mana_agent.coding.models import AgentEvent, CodingTask, CodingTaskResult, WorkspaceContext


@runtime_checkable
class CodingAgentBackend(Protocol):
    name: str

    async def start(self) -> None: ...

    async def execute(self, task: CodingTask, workspace: WorkspaceContext) -> CodingTaskResult: ...

    def stream(self, task: CodingTask, workspace: WorkspaceContext) -> AsyncIterator[AgentEvent]: ...

    async def cancel(self, task_id: str) -> None: ...

    async def close(self) -> None: ...


__all__ = ["CodingAgentBackend"]
