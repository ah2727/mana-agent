from __future__ import annotations

from threading import Lock


class LockTable:
    def __init__(self) -> None:
        self._guard = Lock()
        self._locks: dict[str, Lock] = {}

    def lock_for(self, key: str) -> Lock:
        with self._guard:
            if key not in self._locks:
                self._locks[key] = Lock()
            return self._locks[key]
