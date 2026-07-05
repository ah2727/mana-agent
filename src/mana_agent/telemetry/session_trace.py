from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from mana_agent.cli.events import ChatEvent


@dataclass(slots=True)
class SessionTrace:
    session_id: str
    events: list[ChatEvent] = field(default_factory=list)
    trace_mode: str = "compact"
    path: Path | None = None

    def record(self, event: ChatEvent) -> ChatEvent:
        self.events.append(event)
        if self.path is not None:
            target = Path(self.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.as_dict(), ensure_ascii=False) + "\n")
        return event

    def extend(self, events: Iterable[ChatEvent]) -> None:
        for event in events:
            self.record(event)

    def latest(self, limit: int = 20) -> list[ChatEvent]:
        return self.events[-max(1, int(limit)) :]

    def by_type(self, event_type: str) -> list[ChatEvent]:
        return [event for event in self.events if event.type == event_type]

    def as_jsonl(self) -> str:
        return "\n".join(json.dumps(event.as_dict(), ensure_ascii=False) for event in self.events)

    def write_jsonl(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.as_jsonl() + ("\n" if self.events else ""), encoding="utf-8")
        return target
