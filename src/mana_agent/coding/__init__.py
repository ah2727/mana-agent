"""Provider-neutral coding backend contracts."""

from mana_agent.coding.backend import CodingAgentBackend
from mana_agent.coding.models import (
    AgentEvent,
    CodingBackendDecision,
    CodingTask,
    CodingTaskResult,
    WorkspaceContext,
)
from mana_agent.coding.orchestrator import CodingBackendOrchestrator
from mana_agent.coding.registry import CodingBackendRegistry

__all__ = [
    "AgentEvent",
    "CodingAgentBackend",
    "CodingBackendDecision",
    "CodingBackendOrchestrator",
    "CodingBackendRegistry",
    "CodingTask",
    "CodingTaskResult",
    "WorkspaceContext",
]
