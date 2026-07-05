from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from threading import Lock

_LOCK = Lock()
_COUNTERS: defaultdict[str, int] = defaultdict(int)


def _next(key: str, prefix: str, width: int = 6) -> str:
    with _LOCK:
        _COUNTERS[key] += 1
        value = _COUNTERS[key]
    return f"{prefix}_{value:0{width}d}"


def _role_slug(role: str) -> str:
    return "_".join(str(role or "agent").strip().lower().replace("-", "_").split()) or "agent"


def new_agent_id(role: str) -> str:
    slug = _role_slug(role)
    return _next(f"agent:{slug}", f"agent_{slug}", width=4)


def new_subagent_id(role: str) -> str:
    slug = _role_slug(role)
    return _next(f"subagent:{slug}", f"subagent_{slug}", width=4)


def new_task_id() -> str:
    date = datetime.now().strftime("%Y%m%d")
    return _next(f"task:{date}", f"task_{date}", width=6)


def new_queue_job_id() -> str:
    return _next("queue_job", "queue_job", width=6)


def new_message_id() -> str:
    return _next("message", "message", width=6)


def new_discussion_id() -> str:
    return _next("discussion", "discussion", width=6)


def new_decision_id() -> str:
    return _next("decision", "decision", width=6)


def new_trace_id() -> str:
    return _next("trace", "trace", width=6)
