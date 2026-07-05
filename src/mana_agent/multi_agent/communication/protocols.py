from __future__ import annotations

from typing import Protocol


class AgentProtocol(Protocol):
    agent_id: str

    def run(self, task_id: str, context: dict) -> object: ...
