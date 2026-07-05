from __future__ import annotations

from mana_agent.multi_agent.agents.base_agent import BaseAgent
from mana_agent.multi_agent.core.types import QueueJobType
from mana_agent.multi_agent.queue.queue_manager import QueueManager


class CodingAgent(BaseAgent):
    def __init__(self, *args, queue_manager: QueueManager | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.queue_manager = queue_manager

    def execute_tool_directly(self, *args, **kwargs):
        raise PermissionError("CodingAgent must use QueueManager and cannot execute tools directly")

    def request_read(self, task_id: str, path: str):
        if self.queue_manager is None:
            self.taskboard.add_blocker(task_id, "QueueManager unavailable for coding agent read request")
            return None
        return self.queue_manager.enqueue(
            task_id=task_id,
            requested_by_agent_id=self.agent_id,
            job_type=QueueJobType.REPO_READ,
            payload={"path": path},
            priority=50,
        )

    def request_patch(self, task_id: str, patch: str):
        if self.queue_manager is None:
            self.taskboard.add_blocker(task_id, "QueueManager unavailable for coding agent patch request")
            return None
        return self.queue_manager.enqueue(
            task_id=task_id,
            requested_by_agent_id=self.agent_id,
            job_type=QueueJobType.APPLY_PATCH,
            payload={"patch": patch},
            priority=10,
            lock_key="repo",
            requires_write_lock=True,
        )
