from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from mana_agent.telemetry.tokens import TokenUsage


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ChatEvent:
    event_id: str = field(default_factory=lambda: f"evt-{uuid.uuid4().hex}")
    parent_event_id: str | None = None
    session_id: str = ""
    turn_id: str = ""
    agent_id: str | None = "main"
    subagent_id: str | None = None
    step_id: str | None = None
    type: str = "step.updated"
    status: str = "running"
    title: str = ""
    summary: str | None = None
    started_at: str = field(default_factory=utc_now_iso)
    ended_at: str | None = None
    duration_ms: int | None = None
    token_usage: TokenUsage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def message(self) -> str:
        return self.summary or ""

    @message.setter
    def message(self, value: str | None) -> None:
        self.summary = str(value or "") or None

    def finish(self, *, status: str = "success", message: str | None = None) -> "ChatEvent":
        self.status = status
        if message is not None:
            self.summary = message
        self.ended_at = utc_now_iso()
        try:
            started = datetime.fromisoformat(self.started_at.replace("Z", "+00:00"))
            ended = datetime.fromisoformat(self.ended_at.replace("Z", "+00:00"))
            self.duration_ms = int(max(0.0, (ended - started).total_seconds() * 1000))
        except Exception:
            self.duration_ms = 0
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "parent_event_id": self.parent_event_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "agent_id": self.agent_id,
            "subagent_id": self.subagent_id,
            "step_id": self.step_id,
            "type": self.type,
            "status": self.status,
            "title": self.title,
            "summary": self.summary,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "token_usage": self.token_usage.as_dict() if self.token_usage is not None else None,
            "metadata": dict(self.metadata),
        }


def make_event(
    event_type: str,
    *,
    title: str,
    message: str = "",
    status: str = "running",
    session_id: str = "",
    turn_id: str = "",
    agent_id: str | None = "main",
    subagent_id: str | None = None,
    step_id: str | None = None,
    parent_event_id: str | None = None,
    token_usage: TokenUsage | None = None,
    metadata: dict[str, Any] | None = None,
) -> ChatEvent:
    return ChatEvent(
        parent_event_id=parent_event_id,
        session_id=session_id,
        turn_id=turn_id,
        agent_id=agent_id,
        subagent_id=subagent_id,
        step_id=step_id,
        type=event_type,
        status=status,
        title=title,
        summary=message or None,
        token_usage=token_usage,
        metadata=dict(metadata or {}),
    )
