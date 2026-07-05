from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MultiAgentState:
    active_task_ids: list[str] = field(default_factory=list)
    active_agent_ids: list[str] = field(default_factory=list)
