"""Persist and summarize coding-agent flow memory.

This service stores per-project coding-flow state in
``.mana_index/chat_memory.sqlite3`` and exposes helpers used by the coding agent
to:

- create/resume/reset active flows
- persist turns, extracted tasks, and decisions
- summarize recent context for follow-up turns
- enforce heuristics around patch-loop retries and conflicting requests
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PLAN_TRIGGER_REQUEST_RE = re.compile(
    r"(?i)\b(?:implement|execute|run|apply|trigger)\s+(?:the\s+|last\s+|that\s+|current\s+)?plan\b"
)


def _utc_now() -> str:
    """Return a stable UTC timestamp used across flow persistence rows."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(slots=True)
class FlowSummary:
    """Aggregated view of a persisted coding flow used by UI/prompt context."""

    flow_id: str
    objective: str
    updated_at: str
    constraints: list[str]
    acceptance: list[str]
    open_tasks: list[str]
    recent_decisions: list[dict[str, str]]
    last_changed_files: list[str]
    unresolved_static_findings: list[str]
    checklist: dict[str, Any] | None
    transitions: list[dict[str, Any]]
    last_blocked_reason: str


class CodingMemoryService:
    """SQLite-backed persistence for coding-agent flow continuity."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        max_turns: int = 5,
        max_tasks: int = 20,
    ) -> None:
        """Initialize persistence under ``project_root/.mana_index``."""
        self.project_root = Path(project_root).resolve()
        self.max_turns = max(1, int(max_turns))
        self.max_tasks = max(1, int(max_tasks))
        self.db_path = self.project_root / ".mana_index" / "chat_memory.sqlite3"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Create/upgrade tables needed for flow, turns, tasks, and decisions."""
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS coding_flows (
                    flow_id TEXT PRIMARY KEY,
                    project_root TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    constraints_json TEXT NOT NULL,
                    acceptance_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS coding_flow_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    flow_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    user_request TEXT NOT NULL,
                    effective_prompt TEXT NOT NULL,
                    agent_answer TEXT NOT NULL,
                    changed_files_json TEXT NOT NULL,
                    warnings_json TEXT NOT NULL,
                    static_findings_json TEXT NOT NULL,
                    checklist_json TEXT NOT NULL DEFAULT '{}',
                    transitions_json TEXT NOT NULL DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS coding_flow_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    flow_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    rationale TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS coding_flow_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    flow_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    task_text TEXT NOT NULL,
                    state TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS coding_flow_checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    flow_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_coding_flows_status_updated
                  ON coding_flows(status, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_coding_turns_flow_created
                  ON coding_flow_turns(flow_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_coding_tasks_flow_created
                  ON coding_flow_tasks(flow_id, created_at DESC);
                """
            )
            cols = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(coding_flow_turns)").fetchall()
            }
            if "checklist_json" not in cols:
                conn.execute(
                    "ALTER TABLE coding_flow_turns ADD COLUMN checklist_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "transitions_json" not in cols:
                conn.execute(
                    "ALTER TABLE coding_flow_turns ADD COLUMN transitions_json TEXT NOT NULL DEFAULT '[]'"
                )

    @staticmethod
    def _loads_list(value: str | None) -> list[str]:
        """Parse a JSON list into non-empty string items."""
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except Exception:
            return []
        if not isinstance(parsed, list):
            return []
        return [str(item).strip() for item in parsed if str(item).strip()]

    @staticmethod
    def _objective_from_request(request: str) -> str:
        """Derive a concise objective string from a user request."""
        cleaned = " ".join((request or "").strip().split())
        if not cleaned:
            return "Coding task"
        return cleaned[:200]

    @staticmethod
    def _extract_constraints(request: str) -> list[str]:
        """Extract likely constraint bullets from a request."""
        lines = [line.strip("- ").strip() for line in (request or "").splitlines()]
        constraints: list[str] = []
        signals = ("only ", "do not", "don't ", "without ", "scope ", "no ")
        for line in lines:
            lower = line.lower()
            if any(token in lower for token in signals):
                constraints.append(line[:200])
        return constraints[:8]

    @staticmethod
    def _extract_acceptance(request: str) -> list[str]:
        """Extract acceptance-style signals from a request."""
        lines = [line.strip("- ").strip() for line in (request or "").splitlines()]
        acceptance: list[str] = []
        signals = ("success", "done when", "accept", "should ")
        for line in lines:
            lower = line.lower()
            if any(token in lower for token in signals):
                acceptance.append(line[:200])
        return acceptance[:8]

    @staticmethod
    def _extract_tasks(answer: str) -> tuple[list[str], list[str]]:
        """Extract done/open checkbox tasks from a markdown-like answer."""
        done: list[str] = []
        open_tasks: list[str] = []
        for raw in (answer or "").splitlines():
            line = raw.strip()
            if line.startswith("- [x] ") or line.startswith("* [x] "):
                done.append(line[6:].strip()[:200])
            elif line.startswith("- [ ] ") or line.startswith("* [ ] "):
                open_tasks.append(line[6:].strip()[:200])
        return done, open_tasks

    @staticmethod
    def _extract_decisions(answer: str, warnings: list[str]) -> list[dict[str, str]]:
        """Extract decision/rationale rows from answer text and warning heuristics."""
        rows: list[dict[str, str]] = []
        for raw in (answer or "").splitlines():
            line = raw.strip()
            if line.lower().startswith("decision:"):
                rows.append(
                    {
                        "decision": line.split(":", 1)[1].strip()[:200],
                        "rationale": "Provided in agent answer",
                    }
                )
        for warning in warnings:
            lowered = warning.lower()
            if "write_file fallback" in lowered:
                rows.append(
                    {
                        "decision": "Use write_file fallback",
                        "rationale": warning[:220],
                    }
                )
            elif "patch-only loop" in lowered:
                rows.append(
                    {
                        "decision": "Stop patch-only retries",
                        "rationale": warning[:220],
                    }
                )
        deduped: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for row in rows:
            key = (row["decision"], row["rationale"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        return deduped[:10]

    def get_active_flow_id(self) -> str | None:
        """Return the most recently updated active flow for this project."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT flow_id
                FROM coding_flows
                WHERE project_root = ? AND status = 'active'
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (str(self.project_root),),
            ).fetchone()
        if row is None:
            return None
        return str(row["flow_id"])

    def ensure_flow(self, *, flow_id: str | None, request: str) -> str:
        """Resume an existing flow or create a new active flow for a request."""
        existing = flow_id or self.get_active_flow_id()
        now = _utc_now()
        if existing:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT flow_id FROM coding_flows WHERE flow_id = ?",
                    (existing,),
                ).fetchone()
                if row is not None:
                    conn.execute(
                        "UPDATE coding_flows SET updated_at = ?, status = 'active' WHERE flow_id = ?",
                        (now, existing),
                    )
                    return existing
        created = flow_id or uuid.uuid4().hex[:12]
        objective = self._objective_from_request(request)
        constraints = self._extract_constraints(request)
        acceptance = self._extract_acceptance(request)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO coding_flows (
                    flow_id, project_root, created_at, updated_at, status, objective, constraints_json, acceptance_json
                ) VALUES (?, ?, ?, ?, 'active', ?, ?, ?)
                """,
                (
                    created,
                    str(self.project_root),
                    now,
                    now,
                    objective,
                    json.dumps(constraints),
                    json.dumps(acceptance),
                ),
            )
        return created

    def reset_flow(self, flow_id: str) -> None:
        """Mark a flow as reset so it is no longer treated as active."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE coding_flows SET status = 'reset', updated_at = ? WHERE flow_id = ?",
                (_utc_now(), flow_id),
            )

    def has_prior_patch_failures(self, flow_id: str) -> bool:
        """Check recent warnings for prior patch-loop failure signals."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT warnings_json
                FROM coding_flow_turns
                WHERE flow_id = ?
                ORDER BY created_at DESC
                LIMIT 3
                """,
                (flow_id,),
            ).fetchall()
        for row in rows:
            warnings = self._loads_list(str(row["warnings_json"]))
            for warning in warnings:
                lowered = warning.lower()
                if "patch-style retry" in lowered or "patch-only loop" in lowered:
                    return True
        return False

    def record_turn(
        self,
        *,
        flow_id: str,
        user_request: str,
        effective_prompt: str,
        agent_answer: str,
        changed_files: list[str],
        warnings: list[str],
        static_findings: list[Any],
        checklist: dict[str, Any] | None = None,
        transitions: list[dict[str, Any]] | None = None,
    ) -> None:
        """Persist a completed turn and derived tasks/decisions."""
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO coding_flow_turns (
                    flow_id, created_at, user_request, effective_prompt, agent_answer,
                    changed_files_json, warnings_json, static_findings_json, checklist_json, transitions_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    flow_id,
                    now,
                    user_request,
                    effective_prompt,
                    agent_answer,
                    json.dumps(changed_files),
                    json.dumps(warnings),
                    json.dumps(static_findings),
                    json.dumps(checklist or {}),
                    json.dumps(transitions or []),
                ),
            )
            conn.execute(
                "UPDATE coding_flows SET updated_at = ?, status = 'active' WHERE flow_id = ?",
                (now, flow_id),
            )

            done, open_tasks = self._extract_tasks(agent_answer)
            for task in done[: self.max_tasks]:
                conn.execute(
                    """
                    INSERT INTO coding_flow_tasks (flow_id, created_at, task_text, state)
                    VALUES (?, ?, ?, 'done')
                    """,
                    (flow_id, now, task),
                )
            for task in open_tasks[: self.max_tasks]:
                conn.execute(
                    """
                    INSERT INTO coding_flow_tasks (flow_id, created_at, task_text, state)
                    VALUES (?, ?, ?, 'open')
                    """,
                    (flow_id, now, task),
                )

            for decision in self._extract_decisions(agent_answer, warnings):
                conn.execute(
                    """
                    INSERT INTO coding_flow_decisions (flow_id, created_at, decision, rationale)
                    VALUES (?, ?, ?, ?)
                    """,
                    (flow_id, now, decision["decision"], decision["rationale"]),
                )

    def persist_preview_checklist(
        self,
        *,
        flow_id: str,
        user_request: str,
        checklist: dict[str, Any],
        source: str,
        warning: str = "",
    ) -> None:
        """Persist a planner preview checklist so flow views match pre-execution UI."""
        now = _utc_now()
        warnings = [warning] if warning.strip() else []
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO coding_flow_turns (
                    flow_id, created_at, user_request, effective_prompt, agent_answer,
                    changed_files_json, warnings_json, static_findings_json, checklist_json, transitions_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    flow_id,
                    now,
                    user_request,
                    f"preview_checklist_source={source}",
                    "Preview checklist generated.",
                    json.dumps([]),
                    json.dumps(warnings),
                    json.dumps([]),
                    json.dumps(checklist or {}),
                    json.dumps(
                        [
                            {
                                "from_phase": "preview",
                                "to_phase": "preview",
                                "reason": f"prechecklist_source={source}",
                            }
                        ]
                    ),
                ),
            )
            conn.execute(
                "UPDATE coding_flows SET updated_at = ?, status = 'active' WHERE flow_id = ?",
                (now, flow_id),
            )

    def build_flow_context(self, flow_id: str, repo_delta_paths: list[str]) -> str:
        """Build a prompt-ready textual summary of the current flow state."""
        summary = self.get_flow_summary(flow_id)
        if summary is None:
            return ""
        parts: list[str] = []
        parts.append(f"Flow ID: {summary.flow_id}")
        parts.append(f"Current objective: {summary.objective}")
        if summary.constraints:
            parts.append("Locked constraints:")
            for item in summary.constraints:
                parts.append(f"- {item}")
        if summary.acceptance:
            parts.append("Acceptance criteria:")
            for item in summary.acceptance:
                parts.append(f"- {item}")
        if summary.open_tasks:
            parts.append("Open tasks:")
            for item in summary.open_tasks:
                parts.append(f"- [ ] {item}")
        if summary.recent_decisions:
            parts.append("Recent decisions:")
            for item in summary.recent_decisions:
                parts.append(f"- {item['decision']} ({item['rationale']})")
        if summary.checklist:
            parts.append("Current checklist:")
            steps = summary.checklist.get("steps", []) if isinstance(summary.checklist, dict) else []
            for step in steps[:20]:
                if isinstance(step, dict):
                    status = str(step.get("status", "pending"))
                    title = str(step.get("title", "step"))
                    parts.append(f"- [{status}] {title}")
            if summary.last_blocked_reason:
                parts.append(f"Last blocked reason: {summary.last_blocked_reason}")
        if summary.last_changed_files:
            parts.append("Last changed files:")
            for item in summary.last_changed_files[:20]:
                parts.append(f"- {item}")
        if summary.unresolved_static_findings:
            parts.append("Unresolved static findings:")
            for item in summary.unresolved_static_findings[:10]:
                parts.append(f"- {item}")
        if repo_delta_paths:
            parts.append("Current repository delta paths:")
            for path in repo_delta_paths[:40]:
                parts.append(f"- {path}")
        return "\n".join(parts).strip()

    def checkpoint(self, flow_id: str, snapshot: dict[str, Any] | None = None) -> None:
        """Persist a flow checkpoint snapshot for later inspection/debugging."""
        payload = snapshot or {}
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO coding_flow_checkpoints (flow_id, created_at, snapshot_json)
                VALUES (?, ?, ?)
                """,
                (flow_id, _utc_now(), json.dumps(payload)),
            )

    def get_flow_summary(self, flow_id: str) -> FlowSummary | None:
        """Return aggregated flow state from recent turns/tasks/decisions."""
        with self._connect() as conn:
            flow_row = conn.execute(
                """
                SELECT flow_id, objective, updated_at, constraints_json, acceptance_json
                FROM coding_flows
                WHERE flow_id = ?
                LIMIT 1
                """,
                (flow_id,),
            ).fetchone()
            if flow_row is None:
                return None

            task_rows = conn.execute(
                """
                SELECT task_text, state
                FROM coding_flow_tasks
                WHERE flow_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (flow_id, self.max_tasks * 3),
            ).fetchall()
            task_state: dict[str, str] = {}
            for row in task_rows:
                task = str(row["task_text"])
                if task not in task_state:
                    task_state[task] = str(row["state"])
            open_tasks = [task for task, state in task_state.items() if state == "open"][: self.max_tasks]

            decision_rows = conn.execute(
                """
                SELECT decision, rationale
                FROM coding_flow_decisions
                WHERE flow_id = ?
                ORDER BY id DESC
                LIMIT 6
                """,
                (flow_id,),
            ).fetchall()
            recent_decisions = [
                {"decision": str(row["decision"]), "rationale": str(row["rationale"])}
                for row in reversed(decision_rows)
            ]

            turn_rows = conn.execute(
                """
                SELECT changed_files_json, static_findings_json, checklist_json, transitions_json
                FROM coding_flow_turns
                WHERE flow_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (flow_id, self.max_turns),
            ).fetchall()

        last_changed_files: list[str] = []
        unresolved: list[str] = []
        latest_checklist: dict[str, Any] | None = None
        latest_transitions: list[dict[str, Any]] = []
        last_blocked_reason = ""
        for row in turn_rows:
            for path in self._loads_list(str(row["changed_files_json"])):
                if path not in last_changed_files:
                    last_changed_files.append(path)
            try:
                findings = json.loads(str(row["static_findings_json"]))
            except Exception:
                findings = []
            if isinstance(findings, list):
                for item in findings:
                    text = str(item)
                    if text and text not in unresolved:
                        unresolved.append(text[:220])
            if latest_checklist is None:
                try:
                    raw_checklist = json.loads(str(row["checklist_json"]))
                    if isinstance(raw_checklist, dict) and raw_checklist:
                        latest_checklist = raw_checklist
                except Exception:
                    pass
            if not latest_transitions:
                try:
                    raw_transitions = json.loads(str(row["transitions_json"]))
                    if isinstance(raw_transitions, list):
                        latest_transitions = [item for item in raw_transitions if isinstance(item, dict)]
                except Exception:
                    pass
        for item in reversed(latest_transitions):
            if str(item.get("to_phase", "")) == "blocked":
                last_blocked_reason = str(item.get("reason", "")).strip()
                if last_blocked_reason:
                    break

        return FlowSummary(
            flow_id=str(flow_row["flow_id"]),
            objective=str(flow_row["objective"]),
            updated_at=str(flow_row["updated_at"]),
            constraints=self._loads_list(str(flow_row["constraints_json"])),
            acceptance=self._loads_list(str(flow_row["acceptance_json"])),
            open_tasks=open_tasks,
            recent_decisions=recent_decisions,
            last_changed_files=last_changed_files,
            unresolved_static_findings=unresolved,
            checklist=latest_checklist,
            transitions=latest_transitions,
            last_blocked_reason=last_blocked_reason,
        )

    def list_recent_turns(self, flow_id: str) -> list[dict[str, Any]]:
        """List recent turns with changed files and warnings."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT created_at, user_request, changed_files_json, warnings_json
                FROM coding_flow_turns
                WHERE flow_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (flow_id, self.max_turns),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            result.append(
                {
                    "created_at": str(row["created_at"]),
                    "user_request": str(row["user_request"]),
                    "changed_files": self._loads_list(str(row["changed_files_json"])),
                    "warnings": self._loads_list(str(row["warnings_json"])),
                }
            )
        return list(reversed(result))

    def is_conflicting_request(self, flow_id: str, request: str) -> bool:
        """Detect likely request divergence from the active objective."""
        summary = self.get_flow_summary(flow_id)
        if summary is None:
            return False
        if _PLAN_TRIGGER_REQUEST_RE.search(request or ""):
            return False
        objective_words = {w for w in summary.objective.lower().split() if len(w) > 3}
        request_words = {w for w in (request or "").lower().split() if len(w) > 3}
        if not objective_words or not request_words:
            return False
        overlap = len(objective_words.intersection(request_words))
        # low overlap and edit-intent generally indicates switching tracks
        looks_edit = any(token in request.lower() for token in ("fix", "implement", "update", "change", "add"))
        return looks_edit and overlap == 0
