from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentMemory:
    summaries: list[str] = field(default_factory=list)

    def remember(self, summary: str) -> None:
        text = str(summary or "").strip()
        if text and "secret" not in text.lower():
            self.summaries.append(text[:1200])
