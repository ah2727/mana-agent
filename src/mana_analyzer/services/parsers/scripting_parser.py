from __future__ import annotations

import re
from pathlib import Path

from mana_analyzer.services.parsers.base import ParsedModule

_SHELL_FUNC_RE = re.compile(r"^\s*(?:function\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\(\)\s*\{", re.MULTILINE)
_RUBY_DEF_RE = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_!?=]*)", re.MULTILINE)
_PHP_FUNC_RE = re.compile(r"function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def parse_scripting_module(file_path: Path, _project_root: Path) -> ParsedModule:
    source = file_path.read_text(encoding="utf-8", errors="ignore")
    suffix = file_path.suffix.lower()
    parsed = ParsedModule(parse_mode="full")

    if suffix in {".sh", ".bash", ".zsh"}:
        parsed.functions.extend(sorted(set(_SHELL_FUNC_RE.findall(source))))
    elif suffix == ".rb":
        parsed.functions.extend(sorted(set(_RUBY_DEF_RE.findall(source))))
    elif suffix == ".php":
        parsed.functions.extend(sorted(set(_PHP_FUNC_RE.findall(source))))
    elif suffix == ".sql":
        parsed.functions.extend(sorted(set(re.findall(r"\b(create\s+table|create\s+view|create\s+function)\b", source, flags=re.IGNORECASE))))

    return parsed
