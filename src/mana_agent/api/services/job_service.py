from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from mana_agent.workspaces.paths import mana_home
from mana_agent.workspaces.store import atomic_write_json


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ApiJobStore:
    def __init__(self) -> None:
        self.directory = mana_home() / "api" / "jobs"

    def create(self, kind: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        job = {
            "job_id": f"job_{uuid.uuid4().hex[:20]}",
            "kind": kind,
            "status": "queued",
            "metadata": metadata or {},
            "result": None,
            "error": "",
            "created_at": _now(),
            "updated_at": _now(),
        }
        self.save(job)
        return job

    def path(self, job_id: str) -> Path:
        if not job_id.startswith("job_") or not job_id.replace("_", "").isalnum():
            raise ValueError("invalid job id")
        return self.directory / f"{job_id}.json"

    def save(self, job: dict[str, Any]) -> None:
        job["updated_at"] = _now()
        atomic_write_json(self.path(str(job["job_id"])), job)

    def get(self, job_id: str) -> dict[str, Any]:
        import json

        return json.loads(self.path(job_id).read_text(encoding="utf-8"))

    def run(self, job_id: str, operation: Callable[[], dict[str, Any]]) -> None:
        job = self.get(job_id)
        job["status"] = "running"
        self.save(job)
        try:
            job["result"] = operation()
            job["status"] = "done"
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = str(exc)
        self.save(job)
