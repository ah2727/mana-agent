"""Map Codex app-server notifications into Mana-Agent coding events."""

from __future__ import annotations

from typing import Any

from mana_agent.coding.models import AgentEvent


_EVENT_TYPES = {
    "thread/started": "codex.thread.started",
    "turn/started": "codex.turn.started",
    "turn/completed": "codex.turn.completed",
    "turn/failed": "codex.worker.failed",
    "turn/cancelled": "codex.worker.cancelled",
    "item/started": "codex.item.started",
    "item/completed": "codex.item.completed",
    "item/agentMessage/delta": "codex.reasoning.progress",
    "item/commandExecution/outputDelta": "codex.command.progress",
    "approval/requestApproval": "codex.approval.required",
    "item/commandExecution/requestApproval": "codex.approval.required",
    "item/fileChange/requestApproval": "codex.approval.required",
    "item/permissions/requestApproval": "codex.approval.required",
    "execCommandApproval": "codex.approval.required",
    "applyPatchApproval": "codex.approval.required",
}


def adapt_codex_event(task_id: str, notification: dict[str, Any]) -> AgentEvent:
    method = str(notification.get("method") or "")
    params = notification.get("params")
    payload = dict(params) if isinstance(params, dict) else {}
    thread_id = str(payload.get("threadId") or payload.get("thread_id") or "")
    turn = payload.get("turn")
    turn_id = str(payload.get("turnId") or (turn.get("id") if isinstance(turn, dict) else "") or "")
    status = "running"
    if method == "turn/completed":
        status = "success"
    elif method == "turn/failed":
        status = "failed"
    elif method == "turn/cancelled":
        status = "cancelled"
    summary = _summary(payload)
    return AgentEvent(
        event_type=_EVENT_TYPES.get(method, f"codex.{method.replace('/', '.') or 'notification'}"),
        task_id=task_id,
        status=status,  # type: ignore[arg-type]
        title=method,
        summary=summary,
        thread_id=thread_id,
        turn_id=turn_id,
        payload=payload,
    )


def _summary(payload: dict[str, Any]) -> str:
    for key in ("message", "summary", "delta", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:1000]
    item = payload.get("item")
    if isinstance(item, dict):
        for key in ("text", "message", "command"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:1000]
    return ""


__all__ = ["adapt_codex_event"]
