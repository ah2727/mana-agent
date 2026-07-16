"""Bounded pool of isolated Codex backend instances."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from mana_agent.coding.models import CodingTask, CodingTaskResult, WorkspaceContext
from mana_agent.integrations.codex.backend import CodexCodingBackend
from mana_agent.integrations.codex.config import CodexSettings


@dataclass(slots=True)
class TaskHandle:
    task_id: str
    future: asyncio.Task[CodingTaskResult]

    async def wait(self) -> CodingTaskResult:
        return await self.future


class CodexWorkerPool:
    def __init__(self, settings: CodexSettings) -> None:
        self.settings = settings
        self._semaphore = asyncio.Semaphore(settings.max_workers)
        self._handles: dict[str, TaskHandle] = {}
        self._workers: dict[str, CodexCodingBackend] = {}
        self._scope_condition = asyncio.Condition()
        self._active_scopes: dict[str, frozenset[str]] = {}

    async def submit(self, task: CodingTask, workspace: WorkspaceContext) -> TaskHandle:
        if task.task_id in self._handles and not self._handles[task.task_id].future.done():
            raise ValueError(f"Codex task is already active: {task.task_id}")
        future = asyncio.create_task(self._run(task, workspace), name=f"codex-task-{task.task_id}")
        handle = TaskHandle(task_id=task.task_id, future=future)
        self._handles[task.task_id] = handle
        return handle

    async def cancel(self, task_id: str) -> None:
        worker = self._workers.get(task_id)
        if worker is not None:
            await worker.cancel(task_id)
        handle = self._handles.get(task_id)
        if handle is not None and not handle.future.done():
            handle.future.cancel()

    async def wait_all(self) -> list[CodingTaskResult]:
        handles = list(self._handles.values())
        return list(await asyncio.gather(*(handle.wait() for handle in handles)))

    async def close(self) -> None:
        await asyncio.gather(*(worker.close() for worker in self._workers.values()), return_exceptions=True)
        self._workers.clear()

    async def _run(self, task: CodingTask, workspace: WorkspaceContext) -> CodingTaskResult:
        scope = frozenset(_normalize_scope(task.allowed_files))
        await self._lease_scope(task.task_id, scope)
        try:
            async with self._semaphore:
                worker = CodexCodingBackend(self.settings, worker_id=f"codex-{task.task_id}")
                self._workers[task.task_id] = worker
                try:
                    return await worker.execute(task, workspace)
                finally:
                    await worker.close()
                    self._workers.pop(task.task_id, None)
        finally:
            async with self._scope_condition:
                self._active_scopes.pop(task.task_id, None)
                self._scope_condition.notify_all()

    async def _lease_scope(self, task_id: str, scope: frozenset[str]) -> None:
        async with self._scope_condition:
            await self._scope_condition.wait_for(
                lambda: all(not _scopes_overlap(scope, active) for active in self._active_scopes.values())
            )
            self._active_scopes[task_id] = scope


def _normalize_scope(paths: list[str]) -> list[str]:
    return [str(path).strip().replace("\\", "/").lstrip("./") for path in paths if str(path).strip()]


def _scopes_overlap(left: frozenset[str], right: frozenset[str]) -> bool:
    if not left or not right:
        return True
    for first in left:
        for second in right:
            if first == second or first.startswith(second.rstrip("/") + "/") or second.startswith(first.rstrip("/") + "/"):
                return True
    return False


__all__ = ["CodexWorkerPool", "TaskHandle"]
