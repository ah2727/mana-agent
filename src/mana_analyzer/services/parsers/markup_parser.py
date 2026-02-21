from __future__ import annotations

from pathlib import Path

from mana_analyzer.services.parsers.base import ParsedModule


def parse_markup_module(file_path: Path, _project_root: Path) -> ParsedModule:
    source = file_path.read_text(encoding="utf-8", errors="ignore")
    parsed = ParsedModule(parse_mode="full")

    lines = [line for line in source.splitlines() if line.strip()]
    parsed.constants.append(f"lines:{len(lines)}")
    return parsed
