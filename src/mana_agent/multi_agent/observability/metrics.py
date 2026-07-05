from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Counter:
    name: str
    value: int = 0

    def inc(self, amount: int = 1) -> None:
        self.value += amount
