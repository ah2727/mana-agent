from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from mana_agent.multi_agent.core.types import (
    AgentMessage,
    DecisionRecord,
    DecisionStatus,
    DiscussionStatus,
    DiscussionThread,
    HandoffRecord,
    QueueJob,
    QueueJobStatus,
    QueueJobType,
    RiskLevel,
    TaskBoardItem,
    TaskStatus,
    VerificationResult,
    parse_dt,
    to_jsonable,
)

T = TypeVar("T")


def _enum(enum_cls, value):
    if isinstance(value, enum_cls):
        return value
    return enum_cls(value)


def _dataclass_from_dict(cls: type[T], payload: dict[str, Any]) -> T:
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name in payload:
            kwargs[item.name] = payload[item.name]
    return cls(**kwargs)  # type: ignore[arg-type]


def handoff_from_dict(payload: dict[str, Any]) -> HandoffRecord:
    payload = dict(payload)
    payload["created_at"] = parse_dt(payload.get("created_at"))
    return _dataclass_from_dict(HandoffRecord, payload)


def verification_from_dict(payload: dict[str, Any]) -> VerificationResult:
    payload = dict(payload)
    payload["created_at"] = parse_dt(payload.get("created_at"))
    return _dataclass_from_dict(VerificationResult, payload)


def task_from_dict(payload: dict[str, Any]) -> TaskBoardItem:
    payload = dict(payload)
    payload["status"] = _enum(TaskStatus, payload.get("status", TaskStatus.NEW.value))
    payload["risk_level"] = _enum(RiskLevel, payload.get("risk_level", RiskLevel.LOW.value))
    payload["handoff_records"] = [
        handoff_from_dict(item) for item in payload.get("handoff_records", []) if isinstance(item, dict)
    ]
    payload["verification_results"] = [
        verification_from_dict(item)
        for item in payload.get("verification_results", [])
        if isinstance(item, dict)
    ]
    payload["created_at"] = parse_dt(payload.get("created_at"))
    payload["updated_at"] = parse_dt(payload.get("updated_at"))
    return _dataclass_from_dict(TaskBoardItem, payload)


def message_from_dict(payload: dict[str, Any]) -> AgentMessage:
    from mana_agent.multi_agent.core.types import MessageType

    payload = dict(payload)
    payload["message_type"] = _enum(MessageType, payload.get("message_type"))
    payload["created_at"] = parse_dt(payload.get("created_at"))
    return _dataclass_from_dict(AgentMessage, payload)


def discussion_from_dict(payload: dict[str, Any]) -> DiscussionThread:
    payload = dict(payload)
    payload["status"] = _enum(DiscussionStatus, payload.get("status", DiscussionStatus.OPEN.value))
    payload["created_at"] = parse_dt(payload.get("created_at"))
    payload["updated_at"] = parse_dt(payload.get("updated_at"))
    return _dataclass_from_dict(DiscussionThread, payload)


def decision_from_dict(payload: dict[str, Any]) -> DecisionRecord:
    payload = dict(payload)
    payload["decision_status"] = _enum(DecisionStatus, payload.get("decision_status", DecisionStatus.PROPOSED.value))
    payload["created_at"] = parse_dt(payload.get("created_at"))
    return _dataclass_from_dict(DecisionRecord, payload)


def queue_job_from_dict(payload: dict[str, Any]) -> QueueJob:
    payload = dict(payload)
    payload["job_type"] = _enum(QueueJobType, payload.get("job_type"))
    payload["status"] = _enum(QueueJobStatus, payload.get("status", QueueJobStatus.PENDING.value))
    payload["created_at"] = parse_dt(payload.get("created_at"))
    payload["updated_at"] = parse_dt(payload.get("updated_at"))
    return _dataclass_from_dict(QueueJob, payload)


class JsonStateStore:
    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()
        self.base_dir = self.root / ".mana" / "taskboard"
        self.state_path = self.base_dir / "state.json"
        self.history_path = self.base_dir / "history.jsonl"

    def load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def save_state(self, payload: dict[str, Any]) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(to_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )

    def append_history(self, event: dict[str, Any]) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_jsonable(event), sort_keys=True, ensure_ascii=False) + "\n")


def serialize(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(value)
    return to_jsonable(value)
