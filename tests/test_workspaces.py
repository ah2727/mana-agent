from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from mana_agent.multi_agent.runtime.tool_worker_process import ToolRunRequest, ToolWorkerClient, ToolWorkerProcessError
from mana_agent.services.memory_service import MultiAgentMemoryService
from mana_agent.workspaces.impact import ImpactService
from mana_agent.workspaces.models import WorkspaceSearchRequest
from mana_agent.workspaces.relationships import RelationshipService
from mana_agent.workspaces.routing import RepositoryScopeDecisionEngine
from mana_agent.workspaces.search import WorkspaceSearchService
from mana_agent.workspaces.service import WorkspaceService


def _git(path: Path, *args: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["git", *args], cwd=path, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr


def _repo(path: Path, files: dict[str, str] | None = None) -> Path:
    _git(path, "init")
    _git(path, "config", "user.name", "Mana Test")
    _git(path, "config", "user.email", "mana@example.test")
    for name, content in (files or {"README.md": "test\n"}).items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "-m", "initial")
    return path


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "mana-home"
    monkeypatch.setenv("MANA_HOME", str(home))
    return home


def test_workspace_discovers_repositories_and_preserves_stable_identity(tmp_path: Path, isolated_home: Path) -> None:
    root = tmp_path / "projects"
    first = _repo(root / "api")
    _repo(root / "web")
    service = WorkspaceService()
    workspace = service.create_workspace("product", roots=[root], allowed_roots=[root], discover=True)

    assert len(workspace.repository_ids) == 2
    initial = service.register_repository(first)
    refreshed = service.register_repository(first)
    assert refreshed.repository_id == initial.repository_id
    assert not (first / ".mana").exists()


def test_sessions_and_repository_memory_are_isolated(tmp_path: Path, isolated_home: Path) -> None:
    repo_path = _repo(tmp_path / "repo")
    service = WorkspaceService()
    repo = service.register_repository(repo_path)
    workspace = service.workspace_for_repository(repo.repository_id)
    one = service.create_session(repo_path, workspace_id=workspace.workspace_id)
    two = service.create_session(repo_path, workspace_id=workspace.workspace_id)
    first = MultiAgentMemoryService(
        root=repo_path,
        workspace_id=workspace.workspace_id,
        repository_id=repo.repository_id,
        session_id=one.session_id,
    )
    normalized, fingerprint = first.normalize_task(goal="private session task", repository_ids=[repo.repository_id])
    first.register_task(task_id="task_one", normalized_goal=normalized, fingerprint=fingerprint)
    second = MultiAgentMemoryService(
        root=repo_path,
        workspace_id=workspace.workspace_id,
        repository_id=repo.repository_id,
        session_id=two.session_id,
    )
    assert "task_one" not in second.task_records


def test_cross_repo_search_qualifies_same_relative_path(tmp_path: Path, isolated_home: Path) -> None:
    first_path = _repo(tmp_path / "api", {"src/config.py": "WORKSPACE_NEEDLE = 1\n"})
    second_path = _repo(tmp_path / "worker", {"src/config.py": "WORKSPACE_NEEDLE = 2\n"})
    service = WorkspaceService()
    workspace = service.create_workspace("product", roots=[tmp_path], allowed_roots=[tmp_path])
    first = service.add_repository(workspace.workspace_id, first_path)
    second = service.add_repository(workspace.workspace_id, second_path)

    result = WorkspaceSearchService().search(
        WorkspaceSearchRequest(
            workspace_id=workspace.workspace_id,
            query="WORKSPACE_NEEDLE",
            mode="text",
            repository_ids=[first.repository_id, second.repository_id],
        )
    )
    paths = {item["qualified_path"] for item in result["results"]}
    assert paths == {"api::src/config.py", "worker::src/config.py"}


def test_relationship_and_impact_follow_declared_dependency(tmp_path: Path, isolated_home: Path) -> None:
    library_path = _repo(tmp_path / "shared", {"package.json": json.dumps({"name": "@demo/shared"})})
    app_path = _repo(
        tmp_path / "app",
        {"package.json": json.dumps({"name": "@demo/app", "dependencies": {"@demo/shared": "workspace:*"}})},
    )
    service = WorkspaceService()
    workspace = service.create_workspace("product", roots=[tmp_path], allowed_roots=[tmp_path])
    library = service.add_repository(workspace.workspace_id, library_path)
    app = service.add_repository(workspace.workspace_id, app_path)
    relationships = RelationshipService(service.store).detect(service.store.get_workspace(workspace.workspace_id))
    assert any(item.source_repository_id == app.repository_id and item.target_repository_id == library.repository_id for item in relationships)
    report = ImpactService(service.store).analyze(workspace.workspace_id, library.repository_id, ["src/index.ts"])
    assert [item.repository_id for item in report.affected] == [app.repository_id]


def test_model_scope_selects_multiple_repositories_and_validates_membership(tmp_path: Path, isolated_home: Path) -> None:
    first_path = _repo(tmp_path / "one")
    second_path = _repo(tmp_path / "two")
    service = WorkspaceService()
    workspace = service.create_workspace("product", roots=[tmp_path], allowed_roots=[tmp_path])
    first = service.add_repository(workspace.workspace_id, first_path)
    second = service.add_repository(workspace.workspace_id, second_path)
    session = service.create_session(first_path, workspace_id=workspace.workspace_id)
    context = service.context_for_session(session.session_id)

    class Model:
        def invoke(self, _messages):  # noqa: ANN001
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "repository_ids": [first.repository_id, second.repository_id],
                        "access_by_repository": {first.repository_id: "read", second.repository_id: "read"},
                        "requires_multi_repo": True,
                        "safe_to_continue": True,
                        "reason": "Both repositories implement the requested contract.",
                    }
                )
            )

    decision = RepositoryScopeDecisionEngine(Model()).decide(request="compare contracts", context=context)
    assert decision.repository_ids == [first.repository_id, second.repository_id]


def test_tool_worker_rejects_repository_scope_mismatch(tmp_path: Path, isolated_home: Path) -> None:
    client = ToolWorkerClient(
        api_key="test",
        model="test",
        repo_root=tmp_path,
        project_root=tmp_path,
        workspace_id="workspace_one",
        repository_id="repo_one",
    )
    with pytest.raises(ToolWorkerProcessError, match="repository does not match"):
        client.run_tools(
            ToolRunRequest(
                question="read",
                index_dir=str(tmp_path),
                repository_id="repo_two",
                workspace_id="workspace_one",
            )
        )
