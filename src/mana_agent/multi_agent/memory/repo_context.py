from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RepoContext:
    root: str
    facts: list[str] = field(default_factory=list)
