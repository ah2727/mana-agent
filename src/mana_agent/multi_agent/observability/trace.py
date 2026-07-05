from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mana_agent.multi_agent.core.ids import new_trace_id
from mana_agent.multi_agent.core.types import TraceEvent, to_jsonable


class TraceWriter:
    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()
        self.dir = self.root / ".mana" / "traces"

    def emit(self, event_type: str, *, task_id: str | None = None, agent_id: str | None = None, payload: dict[str, Any] | None = None) -> TraceEvent:
        event = TraceEvent(new_trace_id(), event_type, task_id=task_id, agent_id=agent_id, payload=payload or {})
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{event.trace_id}.json"
        path.write_text(json.dumps(to_jsonable(event), indent=2, sort_keys=True), encoding="utf-8")
        return event
