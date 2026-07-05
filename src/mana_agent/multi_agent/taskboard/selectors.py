from __future__ import annotations

from mana_agent.multi_agent.core.types import TaskBoardItem, TaskStatus


def tasks_by_status(tasks: dict[str, TaskBoardItem], status: TaskStatus) -> list[TaskBoardItem]:
    return [task for task in tasks.values() if task.status == status]


def root_tasks(tasks: dict[str, TaskBoardItem]) -> list[TaskBoardItem]:
    return [task for task in tasks.values() if task.parent_task_id is None]
