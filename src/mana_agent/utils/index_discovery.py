from __future__ import annotations

from pathlib import Path

from mana_agent.config.settings import MANA_ROOT_DIRNAME
from mana_agent.utils.io import EXCLUDED_DIRS
from mana_agent.workspaces.paths import repository_index_dir
from mana_agent.workspaces.service import WorkspaceService


def discover_index_dirs(root_dir: str | Path) -> list[Path]:
    root = Path(root_dir).resolve()
    if root.is_file():
        root = root.parent

    discovered: set[Path] = set()

    # Canonical user-level indexes for the repository's workspace.
    service = WorkspaceService()
    try:
        repo = service.register_repository(root)
        workspace = service.workspace_for_repository(repo.repository_id)
        for repository_id in workspace.repository_ids:
            index_dir = repository_index_dir(repository_id)
            if index_dir.is_dir():
                discovered.add(index_dir.resolve())
    except (OSError, ValueError):
        pass

    # Preferred layout: <project>/.mana/index
    for path in root.rglob(MANA_ROOT_DIRNAME):
        if not path.is_dir():
            continue
        index_dir = path / "index"
        if not index_dir.is_dir():
            continue
        relative_parts = index_dir.relative_to(root).parts
        if any(part in EXCLUDED_DIRS for part in relative_parts if part not in {MANA_ROOT_DIRNAME, "index"}):
            continue
        discovered.add(index_dir.resolve())

    # Backward-compatible layout: <project>/.mana_index
    for path in root.rglob(".mana_index"):
        if not path.is_dir():
            continue
        relative_parts = path.relative_to(root).parts
        if any(part in EXCLUDED_DIRS for part in relative_parts if part != ".mana_index"):
            continue
        discovered.add(path.resolve())

    return sorted(discovered, key=lambda item: str(item))
