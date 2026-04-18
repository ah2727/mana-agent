from __future__ import annotations

from pathlib import Path

from mana_analyzer.config.settings import MANA_ROOT_DIRNAME
from mana_analyzer.utils.io import EXCLUDED_DIRS


def discover_index_dirs(root_dir: str | Path) -> list[Path]:
    root = Path(root_dir).resolve()
    if root.is_file():
        root = root.parent

    discovered: set[Path] = set()

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
