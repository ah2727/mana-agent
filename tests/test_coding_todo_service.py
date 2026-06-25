from __future__ import annotations

from pathlib import Path

import pytest

from mana_agent.services.coding_memory_service import CodingMemoryService
from mana_agent.services.coding_todo_service import TodoService


@pytest.fixture
def todo_service(tmp_path: Path) -> TodoService:
    memory = CodingMemoryService(project_root=tmp_path, max_turns=5, max_tasks=20)
    return TodoService(memory=memory)


def _prechecklist() -> dict:
    return {
        "objective": "add docs",
        "requires_edit": True,
        "target_files": ["docs/05.md"],
        "steps": [
            {"id": "discover", "title": "Locate files"},
            {"id": "read", "title": "Read candidates"},
            {"id": "edit", "title": "Write docs/05.md", "requires_tools": ["write_file"]},
            {"id": "verify", "title": "Verify build", "requires_tools": ["verify_project"]},
        ],
    }


def test_classify_step_uses_tools_then_title() -> None:
    assert TodoService.classify_step({"requires_tools": ["apply_patch"]}) == "edit"
    assert TodoService.classify_step({"requires_tools": ["verify_project"]}) == "verify"
    assert TodoService.classify_step({"title": "Implement the parser"}) == "edit"
    assert TodoService.classify_step({"title": "Verify tests pass"}) == "verify"
    assert TodoService.classify_step({"title": "Locate relevant files"}) == "discover"


def test_sync_from_preview_persists_todos_with_kinds(todo_service: TodoService) -> None:
    flow_id = todo_service.memory.ensure_flow(flow_id=None, request="add docs")
    todos = todo_service.sync_from_preview(flow_id=flow_id, prechecklist=_prechecklist(), source="planner")

    kinds = {t["id"]: t["kind"] for t in todos}
    assert kinds == {"discover": "discover", "read": "read", "edit": "edit", "verify": "verify"}
    assert all(t["status"] == "pending" for t in todos)
    # Ordering preserved.
    assert [t["id"] for t in todos] == ["discover", "read", "edit", "verify"]


def test_reconcile_marks_edit_and_verify_done_on_success(todo_service: TodoService) -> None:
    flow_id = todo_service.memory.ensure_flow(flow_id=None, request="add docs")
    todo_service.sync_from_preview(flow_id=flow_id, prechecklist=_prechecklist())

    todos = todo_service.reconcile_after_run(
        flow_id=flow_id,
        changed_files=["docs/05.md"],
        mutation_succeeded=True,
        verification_passed=True,
        run_blocked=False,
    )
    status = {t["id"]: t["status"] for t in todos}
    assert status == {"discover": "done", "read": "done", "edit": "done", "verify": "done"}


def test_reconcile_blocks_edit_when_run_blocked_without_changes(todo_service: TodoService) -> None:
    flow_id = todo_service.memory.ensure_flow(flow_id=None, request="add docs")
    todo_service.sync_from_preview(flow_id=flow_id, prechecklist=_prechecklist())

    todos = todo_service.reconcile_after_run(
        flow_id=flow_id,
        changed_files=[],
        mutation_succeeded=False,
        verification_passed=False,
        run_blocked=True,
    )
    status = {t["id"]: t["status"] for t in todos}
    assert status["edit"] == "blocked"
    assert status["verify"] == "blocked"
    assert status["discover"] == "done"


def test_sync_is_monotonic_and_prunes_dropped_steps(todo_service: TodoService) -> None:
    flow_id = todo_service.memory.ensure_flow(flow_id=None, request="add docs")
    todo_service.sync_from_preview(flow_id=flow_id, prechecklist=_prechecklist())
    todo_service.memory.update_plan_step_status(flow_id=flow_id, step_id="edit", status="done")

    # Re-preview with a shorter plan: dropped steps pruned, completed step kept.
    smaller = {"steps": [{"id": "edit", "title": "Write docs/05.md", "requires_tools": ["write_file"]}]}
    todos = todo_service.sync_from_preview(flow_id=flow_id, prechecklist=smaller)

    assert [t["id"] for t in todos] == ["edit"]
    # Status does not regress from done back to pending.
    assert todos[0]["status"] == "done"
