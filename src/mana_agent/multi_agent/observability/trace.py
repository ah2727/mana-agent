from __future__ import annotations

import json
from pathlib import Path
from mana_agent.workspaces.paths import mana_home
from typing import Any

from mana_agent.multi_agent.core.ids import new_trace_id
from mana_agent.multi_agent.core.types import ExecutionContext, TraceEvent, enrich_event_identity, to_jsonable


class TraceWriter:
    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()
        self.dir = mana_home() / "traces"

    def emit(
        self,
        event_type: str,
        *,
        task_id: str | None = None,
        agent_id: str | None = None,
        subagent_id: str | None = None,
        agent_role: str | None = None,
        parent_agent_id: str | None = None,
        requested_by_agent_id: str | None = None,
        queue_job_id: str | None = None,
        root_task_id: str | None = None,
        delegation_path: list[str] | None = None,
        context: ExecutionContext | dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> TraceEvent:
        identity = enrich_event_identity(
            {
                "task_id": task_id,
                "agent_id": agent_id,
                "subagent_id": subagent_id,
                "agent_role": agent_role,
                "parent_agent_id": parent_agent_id,
                "requested_by_agent_id": requested_by_agent_id,
                "queue_job_id": queue_job_id,
                "root_task_id": root_task_id,
                "delegation_path": list(delegation_path or []),
            },
            context,
        )
        event = TraceEvent(
            new_trace_id(),
            event_type,
            task_id=identity.get("task_id"),
            agent_id=identity.get("agent_id"),
            subagent_id=identity.get("subagent_id"),
            agent_role=identity.get("agent_role"),
            parent_agent_id=identity.get("parent_agent_id"),
            requested_by_agent_id=identity.get("requested_by_agent_id"),
            queue_job_id=identity.get("queue_job_id"),
            root_task_id=identity.get("root_task_id"),
            delegation_path=list(identity.get("delegation_path") or []),
            payload=payload or {},
        )
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{event.trace_id}.json"
        path.write_text(json.dumps(to_jsonable(event), indent=2, sort_keys=True), encoding="utf-8")
        return event
