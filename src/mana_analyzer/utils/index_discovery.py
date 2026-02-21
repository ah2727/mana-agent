from __future__ import annotations

from pathlib import Path

from mana_analyzer.utils.io import EXCLUDED_DIRS


def discover_index_dirs(root_dir: str | Path) -> list[Path]:
    root = Path(root_dir).resolve()
    if root.is_file():
        root = root.parent

    discovered: list[Path] = []
    for path in root.rglob(".mana_index"):
        if not path.is_dir():
            continue
        relative_parts = path.relative_to(root).parts
        if any(part in EXCLUDED_DIRS for part in relative_parts if part != ".mana_index"):
            continue
        discovered.append(path.resolve())

    return sorted(set(discovered), key=lambda item: str(item))
