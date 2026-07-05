from __future__ import annotations

from typing import Any

from mana_agent.multi_agent.communication.message_bus import MessageBus
from mana_agent.multi_agent.core.types import AgentRole, AgentState, HandoffRecord, MessageType
from mana_agent.multi_agent.registry.agent_registry import AgentRegistry
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard


class BaseAgent:
    def __init__(
        self,
        *,
        agent_id: str,
        role: AgentRole,
        parent_agent_id: str | None,
        capabilities: list[str],
        allowed_tools: list[str] | None = None,
        mailbox: MessageBus,
        taskboard: TaskBoard,
        message_bus: MessageBus,
        registry: AgentRegistry | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.role = role
        self.parent_agent_id = parent_agent_id
        self.capabilities = capabilities
        self.allowed_tools = allowed_tools or []
        self.state = AgentState.IDLE
        self.mailbox = mailbox
        self.taskboard = taskboard
        self.message_bus = message_bus
        self.registry = registry

    def run(self, task_id: str, context: dict[str, Any]) -> Any:
        self.state = AgentState.RUNNING
        self.record_evidence(task_id, f"{self.role.value} agent started")
        self.state = AgentState.DONE
        return context

    def can_handle(self, task) -> bool:
        required = set(getattr(task, "required_capabilities", []) or [])
        return not required or bool(required.intersection(self.capabilities))

    def send_message(self, task_id: str, to_agent_id: str | None, message_type: MessageType, content: str, *, discussion_id: str | None = None):
        return self.message_bus.send(
            task_id=task_id,
            from_agent_id=self.agent_id,
            to_agent_id=to_agent_id,
            message_type=message_type,
            content=content,
            discussion_id=discussion_id,
        )

    def broadcast(self, task_id: str, message_type: MessageType, content: str):
        return self.message_bus.broadcast(task_id, self.agent_id, message_type, content)

    def open_discussion(self, task_id: str, title: str) -> None:
        self.taskboard.add_evidence(task_id, f"{self.role.value} requested discussion: {title}")

    def handoff(self, target_agent_id: str, task_id: str, reason: str) -> HandoffRecord:
        record = HandoffRecord(self.agent_id, target_agent_id, task_id, reason)
        self.taskboard.add_handoff(task_id, record)
        self.send_message(task_id, target_agent_id, MessageType.HANDOFF, reason)
        return record

    def create_subagent(self, role: AgentRole, capabilities: list[str]):
        if self.registry is None:
            self.taskboard.add_blocker("", "Agent registry unavailable for subagent creation")
            return None
        return self.registry.create_subagent(role, self.agent_id, capabilities)

    def record_evidence(self, task_id: str, evidence: str) -> None:
        self.taskboard.add_evidence(task_id, evidence)

    def record_decision(self, task_id: str, decision) -> None:
        self.taskboard.add_decision(task_id, decision)
