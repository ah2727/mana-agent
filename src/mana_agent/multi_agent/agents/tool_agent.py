from __future__ import annotations

from mana_agent.multi_agent.agents.base_agent import BaseAgent
from mana_agent.multi_agent.queue.queue_manager import QueueManager


class ToolAgent(BaseAgent):
    def __init__(self, *args, queue_manager: QueueManager | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.queue_manager = queue_manager

    def run_approved_jobs(self, max_jobs: int | None = None):
        if self.queue_manager is None:
            return []
        return self.queue_manager.run_until_idle(max_jobs=max_jobs)
