"""Normalize a completed Codex turn into a coding task result."""

from __future__ import annotations

from typing import Any

from mana_agent.coding.models import CodingTask, CodingTaskResult, WorkspaceContext


def parse_codex_result(
    *,
    task: CodingTask,
    workspace: WorkspaceContext,
    worker_id: str,
    thread_id: str,
    turn_id: str,
    notifications: list[dict[str, Any]],
    changed_files: list[str],
) -> CodingTaskResult:
    commands: list[str] = []
    tests: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    summary = ""
    usage: dict[str, int] | None = None
    status = "completed"

    for notification in notifications:
        method = str(notification.get("method") or "")
        params = notification.get("params")
        payload = params if isinstance(params, dict) else {}
        item = payload.get("item")
        if isinstance(item, dict):
            item_type = str(item.get("type") or "")
            command = str(item.get("command") or "").strip()
            if command and command not in commands:
                commands.append(command)
                if _is_test_command(command):
                    tests.append(command)
            if item_type in {"agentMessage", "agent_message"}:
                text = str(item.get("text") or item.get("message") or "").strip()
                if text:
                    summary = text
        if method == "warning":
            message = str(payload.get("message") or "").strip()
            if message:
                warnings.append(message)
        if method in {"turn/failed", "error"}:
            status = "failed"
            errors.append(str(payload.get("message") or payload.get("error") or "Codex turn failed"))
        if method == "turn/cancelled":
            status = "cancelled"
        if method == "turn/completed":
            raw_usage = payload.get("usage")
            turn = payload.get("turn")
            if not isinstance(raw_usage, dict) and isinstance(turn, dict):
                raw_usage = turn.get("usage")
            if isinstance(raw_usage, dict):
                usage = {
                    str(key): int(value)
                    for key, value in raw_usage.items()
                    if isinstance(value, int) and not isinstance(value, bool)
                }

    tests_passed = bool(tests) and status == "completed" and not errors
    return CodingTaskResult(
        task_id=task.task_id,
        worker_id=worker_id,
        backend="codex",
        status=status,  # type: ignore[arg-type]
        summary=summary or ("Codex task completed." if status == "completed" else "Codex task did not complete."),
        changed_files=changed_files,
        commands_run=commands,
        tests_run=tests,
        tests_passed=tests_passed,
        warnings=warnings,
        errors=errors,
        branch_name=workspace.branch_name,
        token_usage=usage,
        thread_id=thread_id,
        turn_id=turn_id,
    )


def _is_test_command(command: str) -> bool:
    executable = command.strip().split(maxsplit=1)[0] if command.strip() else ""
    return executable in {"pytest", "tox", "nox", "npm", "pnpm", "yarn", "cargo", "go", "mvn", "gradle"}


__all__ = ["parse_codex_result"]
