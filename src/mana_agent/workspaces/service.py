from __future__ import annotations

import fnmatch
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from mana_agent.utils.project_discovery import MANIFEST_FILENAMES, MANIFEST_GLOBS
from mana_agent.workspaces.context import WorkspaceContext
from mana_agent.workspaces.discovery import discover_git_repositories
from mana_agent.workspaces.models import (
    RepositoryComponent,
    RepositoryRecord,
    RepositoryStatus,
    SessionRecord,
    WorkspaceDiscoveryConfig,
    WorkspaceRecord,
)
from mana_agent.workspaces.paths import repository_dir
from mana_agent.workspaces.store import WorkspaceStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(path: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=path, text=True, capture_output=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _git_root(path: Path) -> Path | None:
    value = _git(path, "rev-parse", "--show-toplevel")
    return Path(value).resolve() if value else None


def _remote(path: Path) -> str | None:
    return _git(path, "remote", "get-url", "origin") or None


def _language_signals(path: Path) -> list[str]:
    suffixes = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".kt": "kotlin",
        ".dart": "dart",
        ".php": "php",
        ".rb": "ruby",
        ".cs": "csharp",
        ".cpp": "cpp",
        ".c": "c",
    }
    counts: dict[str, int] = {}
    ignored = {".git", ".mana", ".venv", "venv", "node_modules", "dist", "build", "vendor"}
    for current, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [item for item in dirs if item not in ignored]
        for filename in files:
            language = suffixes.get(Path(filename).suffix.lower())
            if language:
                counts[language] = counts.get(language, 0) + 1
        if sum(counts.values()) >= 20_000:
            break
    return [name for name, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def _framework_signals(path: Path) -> list[str]:
    signals = {
        "manage.py": "Django",
        "next.config.js": "Next.js",
        "next.config.ts": "Next.js",
        "vite.config.js": "Vite",
        "vite.config.ts": "Vite",
        "nest-cli.json": "NestJS",
        "pubspec.yaml": "Flutter/Dart",
        "Cargo.toml": "Rust/Cargo",
        "go.mod": "Go",
    }
    found: set[str] = set()
    ignored = {".git", ".mana", ".venv", "venv", "node_modules", "dist", "build", "vendor"}
    for _current, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [item for item in dirs if item not in ignored]
        for filename in files:
            framework = signals.get(filename)
            if framework:
                found.add(framework)
    package_json = path / "package.json"
    if package_json.is_file():
        try:
            payload = json.loads(package_json.read_text(encoding="utf-8"))
            deps = set(payload.get("dependencies", {})) | set(payload.get("devDependencies", {}))
            for package, framework in {
                "react": "React",
                "vue": "Vue",
                "@nestjs/core": "NestJS",
                "express": "Express",
            }.items():
                if package in deps:
                    found.add(framework)
        except Exception:
            pass
    return sorted(found)


def _component_kind(root: Path, component: Path) -> str:
    rel = component.relative_to(root).as_posix().lower()
    names = set(rel.split("/"))
    if names & {"docs", "documentation"}:
        return "docs"
    if names & {"infra", "infrastructure", "terraform", "deploy", "k8s", "helm"}:
        return "infrastructure"
    if names & {"apps", "app", "frontend", "web", "mobile"}:
        return "app"
    if names & {"services", "service", "backend", "api"}:
        return "service"
    if names & {"libs", "lib", "packages", "sdk"}:
        return "library"
    return "unknown"


class WorkspaceService:
    def __init__(self, store: WorkspaceStore | None = None) -> None:
        self.store = store or WorkspaceStore()

    def register_repository(
        self,
        path: str | Path,
        *,
        tags: Iterable[str] = (),
        refresh: bool = False,
    ) -> RepositoryRecord:
        requested = Path(path).expanduser().resolve()
        if not requested.is_dir():
            raise ValueError(f"repository path does not exist: {requested}")
        root = _git_root(requested) or requested
        existing = self.store.find_repository_by_path(root)
        if existing is not None and not refresh:
            existing.status.available = root.is_dir()
            existing.status.dirty = bool(_git(root, "status", "--porcelain=v1")) if existing.git_root else False
            existing.branch = _git(root, "branch", "--show-current") or existing.branch
            existing.head_sha = _git(root, "rev-parse", "HEAD") or existing.head_sha
            existing.updated_at = _now()
            self.store.save_repository(existing)
            self._import_legacy_state(existing)
            return existing
        record = existing or RepositoryRecord(name=root.name or "repository", canonical_path=str(root))
        record.name = root.name or record.name
        record.canonical_path = str(root)
        record.git_root = str(_git_root(root)) if _git_root(root) else None
        record.remote_url = _remote(root)
        record.branch = _git(root, "branch", "--show-current") or None
        record.head_sha = _git(root, "rev-parse", "HEAD") or None
        record.languages = _language_signals(root)
        record.frameworks = _framework_signals(root)
        record.tags = sorted(set(record.tags) | {str(item) for item in tags if str(item).strip()})
        record.components = self._components(root)
        record.kind = "monorepo" if len(record.components) > 1 else ("git" if record.git_root else "project")
        record.status = RepositoryStatus(
            available=True,
            dirty=bool(_git(root, "status", "--porcelain=v1")) if record.git_root else False,
            indexed=(repository_dir(record.repository_id) / "index" / "chunks.jsonl").exists(),
            index_stale=True,
        )
        record.updated_at = _now()
        self.store.save_repository(record)
        self._import_legacy_state(record)
        return record

    def _components(self, root: Path) -> list[RepositoryComponent]:
        ignored = {".git", ".mana", ".venv", "venv", "node_modules", "dist", "build", "vendor", "__pycache__"}
        manifests_by_root: dict[Path, list[Path]] = {}
        for current, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [item for item in dirs if item not in ignored]
            current_path = Path(current)
            for filename in files:
                if filename in MANIFEST_FILENAMES or any(fnmatch.fnmatch(filename, pattern) for pattern in MANIFEST_GLOBS):
                    manifests_by_root.setdefault(current_path, []).append(current_path / filename)
        rows: list[RepositoryComponent] = []
        for component_root, manifests in sorted(manifests_by_root.items(), key=lambda item: str(item[0])):
            rel = component_root.relative_to(root).as_posix() or "."
            rows.append(
                RepositoryComponent(
                    name=component_root.name or root.name,
                    relative_path=rel,
                    kind=_component_kind(root, component_root),  # type: ignore[arg-type]
                    languages=_language_signals(component_root),
                    frameworks=_framework_signals(component_root),
                    manifests=[path.relative_to(root).as_posix() for path in sorted(manifests)],
                )
            )
        return rows

    def create_workspace(
        self,
        name: str,
        *,
        roots: Iterable[str | Path] = (),
        discover: bool = False,
        allowed_roots: Iterable[str | Path] = (),
        implicit: bool = False,
    ) -> WorkspaceRecord:
        clean = str(name or "").strip()
        if not clean:
            raise ValueError("workspace name is required")
        discovery_roots = [str(Path(item).expanduser().resolve()) for item in roots]
        record = WorkspaceRecord(
            name=clean,
            discovery=WorkspaceDiscoveryConfig(roots=discovery_roots),
            allowed_roots=[str(Path(item).expanduser().resolve()) for item in allowed_roots],
            implicit=implicit,
        )
        if discover:
            for repo_path in discover_git_repositories(discovery_roots, max_depth=record.discovery.max_depth):
                repo = self.register_repository(repo_path)
                record.repository_ids.append(repo.repository_id)
        record.repository_ids = list(dict.fromkeys(record.repository_ids))
        record.primary_repository_id = record.repository_ids[0] if record.repository_ids else None
        return self.store.save_workspace(record)

    def add_repository(self, workspace_id: str, path: str | Path, *, external: bool = False) -> RepositoryRecord:
        workspace = self.store.get_workspace(workspace_id)
        resolved = Path(path).expanduser().resolve()
        if not external and workspace.discovery.roots:
            if not any(resolved == Path(root) or Path(root) in resolved.parents for root in map(Path, workspace.discovery.roots)):
                raise PermissionError("repository is outside workspace discovery roots; add it explicitly as external")
        repo = self.register_repository(resolved)
        if repo.repository_id not in workspace.repository_ids:
            workspace.repository_ids.append(repo.repository_id)
        workspace.primary_repository_id = workspace.primary_repository_id or repo.repository_id
        workspace.updated_at = _now()
        self.store.save_workspace(workspace)
        return repo

    def remove_repository(self, workspace_id: str, repository_id: str) -> WorkspaceRecord:
        workspace = self.store.get_workspace(workspace_id)
        workspace.repository_ids = [item for item in workspace.repository_ids if item != repository_id]
        if workspace.primary_repository_id == repository_id:
            workspace.primary_repository_id = workspace.repository_ids[0] if workspace.repository_ids else None
        workspace.updated_at = _now()
        return self.store.save_workspace(workspace)

    def discover(self, workspace_id: str) -> list[RepositoryRecord]:
        workspace = self.store.get_workspace(workspace_id)
        added: list[RepositoryRecord] = []
        for path in discover_git_repositories(
            workspace.discovery.roots,
            max_depth=workspace.discovery.max_depth,
            exclude=workspace.discovery.exclude,
        ):
            repo = self.register_repository(path)
            if repo.repository_id not in workspace.repository_ids:
                workspace.repository_ids.append(repo.repository_id)
                added.append(repo)
        workspace.primary_repository_id = workspace.primary_repository_id or (
            workspace.repository_ids[0] if workspace.repository_ids else None
        )
        workspace.updated_at = _now()
        self.store.save_workspace(workspace)
        return added

    def workspace_for_repository(self, repository_id: str) -> WorkspaceRecord:
        matches = [item for item in self.store.list_workspaces() if repository_id in item.repository_ids]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            implicit = [item for item in matches if item.implicit]
            if len(implicit) == 1:
                return implicit[0]
            raise ValueError("repository belongs to multiple workspaces; select one explicitly")
        repo = self.store.get_repository(repository_id)
        return self._standalone_workspace(repo)

    def _standalone_workspace(self, repo: RepositoryRecord) -> WorkspaceRecord:
        workspace = WorkspaceRecord(
            name=f"standalone-{repo.name}",
            repository_ids=[repo.repository_id],
            primary_repository_id=repo.repository_id,
            discovery=WorkspaceDiscoveryConfig(roots=[repo.canonical_path]),
            allowed_roots=[repo.canonical_path],
            implicit=True,
        )
        return self.store.save_workspace(workspace)

    def create_session(
        self,
        cwd: str | Path,
        *,
        workspace_id: str | None = None,
        session_id: str | None = None,
    ) -> SessionRecord:
        repo = self.register_repository(cwd)
        workspace = self.store.get_workspace(workspace_id) if workspace_id else self.workspace_for_repository(repo.repository_id)
        if repo.repository_id not in workspace.repository_ids:
            raise ValueError("session repository is not a member of selected workspace")
        record = SessionRecord(
            session_id=session_id or SessionRecord.model_fields["session_id"].default_factory(),  # type: ignore[misc]
            workspace_id=workspace.workspace_id,
            primary_repository_id=repo.repository_id,
            attached_repository_ids=list(workspace.repository_ids),
            cwd=repo.canonical_path,
        )
        return self.store.save_session(record)

    def context_for_session(self, session_id: str) -> WorkspaceContext:
        session = self.store.get_session(session_id)
        workspace = self.store.get_workspace(session.workspace_id)
        repos = {repo_id: self.store.get_repository(repo_id) for repo_id in session.attached_repository_ids}
        return WorkspaceContext(workspace=workspace, session=session, repositories=repos)

    def archive_session(self, session_id: str) -> SessionRecord:
        record = self.store.get_session(session_id)
        record.status = "archived"
        record.updated_at = _now()
        return self.store.save_session(record)

    def _import_legacy_state(self, repo: RepositoryRecord) -> None:
        if repo.legacy_imported_at:
            return
        source = Path(repo.canonical_path) / ".mana"
        if not source.is_dir():
            repo.legacy_imported_at = _now()
            self.store.save_repository(repo)
            return
        target = repository_dir(repo.repository_id)
        for name in ("index", "analyze"):
            src = source / name
            dst = target / ("analysis" if name == "analyze" else name)
            if src.is_dir() and not dst.exists():
                shutil.copytree(src, dst)
        marker = target / "legacy-import.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps({"source": str(source), "imported_at": _now()}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        repo.legacy_imported_at = _now()
        self.store.save_repository(repo)
