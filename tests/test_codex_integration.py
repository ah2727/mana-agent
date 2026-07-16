from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

import pytest

from mana_agent.coding.models import CodingBackendDecision, CodingTask, WorkspaceContext
from mana_agent.coding.registry import CodingBackendDecisionError, CodingBackendRegistry
from mana_agent.coding.routing_policy import validate_backend_decision
from mana_agent.integrations.codex.backend import CodexCodingBackend
from mana_agent.integrations.codex.config import CodexSettings
from mana_agent.integrations.codex.event_adapter import adapt_codex_event
from mana_agent.integrations.codex.health import check_codex_health
from mana_agent.multi_agent.codex_pool import _scopes_overlap


class _Backend:
    name = "native"

    async def start(self) -> None: ...
    async def execute(self, task, workspace): ...
    async def stream(self, task, workspace):
        if False:
            yield None
    async def cancel(self, task_id: str) -> None: ...
    async def close(self) -> None: ...


class _FakeClient:
    running = True

    def __init__(self, command: tuple[str, ...], *, approval: bool = False) -> None:
        self.command = command
        self.approval = approval
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    async def start(self) -> None:
        return None

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.requests.append((method, params))
        if method == "thread/start":
            return {"thread": {"id": "thread-1"}}
        if method == "turn/start":
            return {"turn": {"id": "turn-1"}}
        return {}

    async def notifications(self, thread_id: str):
        assert thread_id == "thread-1"
        if self.approval:
            yield {"method": "approval/requestApproval", "params": {"threadId": thread_id}}
            return
        yield {
            "method": "item/completed",
            "params": {
                "threadId": thread_id,
                "item": {"type": "commandExecution", "command": "pytest -q"},
            },
        }
        yield {
            "method": "item/completed",
            "params": {
                "threadId": thread_id,
                "item": {"type": "agentMessage", "text": "Implemented the task."},
            },
        }
        yield {
            "method": "turn/completed",
            "params": {"threadId": thread_id, "turn": {"id": "turn-1"}, "usage": {"inputTokens": 10}},
        }

    async def interrupt(self, *, thread_id: str, turn_id: str) -> None:
        return None

    async def close(self) -> None:
        self.closed = True

    async def deny_server_request(self, request: dict[str, Any]) -> None:
        self.requests.append(("deny", request))


def _git_repo(path: Path) -> None:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)


def _task() -> CodingTask:
    return CodingTask(
        task_id="task-1",
        goal="Implement the requested change",
        allowed_files=["src/app.py"],
        acceptance_criteria=["Tests pass"],
        verification_commands=["pytest -q"],
    )


def _workspace(tmp_path: Path) -> WorkspaceContext:
    repository = tmp_path / "repository"
    worktree = tmp_path / "worktree"
    repository.mkdir()
    _git_repo(worktree)
    return WorkspaceContext(repository_path=repository, worktree_path=worktree, branch_name="mana/task-1")


def test_registry_executes_only_model_selected_backend() -> None:
    registry = CodingBackendRegistry()
    backend = _Backend()
    registry.register(backend)
    decision = CodingBackendDecision(
        decision_id="decision-1",
        coding_required=True,
        selected_backend="native",
        estimated_complexity="low",
        requires_repository_write=True,
        reasons=["The model selected the native backend"],
        safe_to_continue=True,
    )
    assert registry.resolve(decision) is backend


def test_registry_does_not_fallback_when_selected_backend_is_missing() -> None:
    registry = CodingBackendRegistry()
    registry.register(_Backend())
    decision = CodingBackendDecision(
        decision_id="decision-2",
        coding_required=True,
        selected_backend="codex",
        estimated_complexity="high",
        requires_repository_write=True,
        reasons=["The model selected Codex"],
        safe_to_continue=True,
    )
    with pytest.raises(CodingBackendDecisionError, match="No fallback backend was executed"):
        registry.resolve(decision)


def test_invalid_backend_decision_stops_safely() -> None:
    with pytest.raises(CodingBackendDecisionError, match="No backend was executed"):
        validate_backend_decision(
            {
                "decision_id": "decision-3",
                "coding_required": True,
                "estimated_complexity": "high",
                "requires_repository_write": True,
                "reasons": ["Coding is required"],
                "safe_to_continue": True,
            }
        )


def test_codex_backend_uses_thread_turn_protocol_and_normalizes_result(tmp_path: Path) -> None:
    fake: _FakeClient | None = None

    def factory(command: tuple[str, ...]) -> _FakeClient:
        nonlocal fake
        fake = _FakeClient(command)
        return fake

    backend = CodexCodingBackend(CodexSettings(enabled=True), client_factory=factory)
    result = asyncio.run(backend.execute(_task(), _workspace(tmp_path)))

    assert result.status == "completed"
    assert result.backend == "codex"
    assert result.tests_run == ["pytest -q"]
    assert result.tests_passed is True
    assert result.thread_id == "thread-1"
    assert fake is not None
    assert [method for method, _params in fake.requests] == ["thread/start", "turn/start"]
    turn_payload = fake.requests[1][1]
    assert turn_payload["cwd"].endswith("worktree")
    assert turn_payload["sandbox"] == "workspaceWrite"
    assert "Do not commit, push, publish" in turn_payload["input"][0]["text"]


def test_codex_backend_does_not_self_approve(tmp_path: Path) -> None:
    backend = CodexCodingBackend(
        CodexSettings(enabled=True),
        client_factory=lambda command: _FakeClient(command, approval=True),
    )
    result = asyncio.run(backend.execute(_task(), _workspace(tmp_path)))
    assert result.status == "failed"
    assert "did not elevate permissions" in result.errors[0]


def test_codex_events_are_mapped_to_mana_event_contract() -> None:
    event = adapt_codex_event(
        "task-1",
        {"method": "turn/completed", "params": {"threadId": "thread-1", "turn": {"id": "turn-1"}}},
    )
    assert event.event_type == "codex.turn.completed"
    assert event.status == "success"
    assert event.thread_id == "thread-1"
    assert event.turn_id == "turn-1"


def test_writing_workspace_must_be_isolated(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    with pytest.raises(ValueError, match="isolated worktree"):
        WorkspaceContext(repository_path=repository, worktree_path=repository)


def test_pool_scope_overlap_is_conservative() -> None:
    assert _scopes_overlap(frozenset(), frozenset({"src/app.py"}))
    assert _scopes_overlap(frozenset({"src"}), frozenset({"src/app.py"}))
    assert not _scopes_overlap(frozenset({"docs"}), frozenset({"src/app.py"}))


def test_disabled_codex_health_is_explicit(tmp_path: Path) -> None:
    report = check_codex_health(CodexSettings(enabled=False), tmp_path)
    assert report.healthy is False
    assert "Codex integration is disabled" in report.errors
