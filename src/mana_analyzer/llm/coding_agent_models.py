from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class FlowStep(BaseModel):
    """Represents a planned step with tooling guidance and execution status."""

    id: str
    title: str
    reason: str
    status: Literal["pending", "in_progress", "done", "blocked"] = "pending"
    requires_tools: list[str] = Field(default_factory=list)


class FlowChecklist(BaseModel):
    """Structured plan capturing objective, constraints, acceptance criteria, and steps."""

    objective: str
    constraints: list[str] = Field(default_factory=list)
    acceptance: list[str] = Field(default_factory=list)
    steps: list[FlowStep] = Field(default_factory=list)
    next_action: str = ""


class ExecutionDecision(BaseModel):
    phase: Literal["discover", "inspect", "edit", "verify", "answer", "blocked"]
    tool_call_allowed: bool
    why: str


class DynamicReadPolicy(BaseModel):
    """LLM-selected read policy for one coding turn (full-auto only)."""

    read_budget: int
    read_line_window: int
    reason: str = ""


class AskAgentLike(Protocol):
    tools: list[Any]

    def ask(self, question: str, **kwargs: Any) -> Any:  # pragma: no cover
        ...


def as_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [as_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): as_jsonable(v) for k, v in obj.items()}
    return obj

