class MultiAgentError(RuntimeError):
    """Base error for the multi-agent runtime."""


class InvalidTaskTransition(MultiAgentError):
    """Raised when a task status transition is invalid."""


class AgentRegistryError(MultiAgentError):
    """Raised when agent hierarchy constraints are violated."""


class ToolPermissionError(MultiAgentError):
    """Raised when a tool request is not permitted."""
