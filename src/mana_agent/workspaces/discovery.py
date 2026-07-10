from __future__ import annotations

import os
from pathlib import Path


DEFAULT_EXCLUDES = {
    ".git",
    ".mana",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "vendor",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}


def discover_git_repositories(
    roots: list[str | Path],
    *,
    max_depth: int = 6,
    exclude: list[str] | None = None,
) -> list[Path]:
    found: set[Path] = set()
    excluded = DEFAULT_EXCLUDES | {str(item) for item in (exclude or [])}
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            continue
        for current, dirs, _files in os.walk(root, followlinks=False):
            current_path = Path(current)
            try:
                depth = len(current_path.relative_to(root).parts)
            except ValueError:
                continue
            dirs[:] = [name for name in dirs if name not in excluded and not (current_path / name).is_symlink()]
            if (current_path / ".git").exists():
                found.add(current_path.resolve())
                dirs[:] = [name for name in dirs if name != ".git"]
            if depth >= max(0, int(max_depth)):
                dirs[:] = []
    return sorted(found, key=lambda path: str(path))
