from __future__ import annotations

import re
from pathlib import Path

from mana_analyzer.analysis.models import ClassDescriptor
from mana_analyzer.services.parsers.base import ParsedModule

_IMPORT_RE = re.compile(r"^\s*#\s*include\s+[<\"]([^>\"]+)[>\"]", re.MULTILINE)
_FUNC_RE = re.compile(r"^\s*(?:[A-Za-z_][A-Za-z0-9_\s:*<>]+)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{", re.MULTILINE)
_CLASS_RE = re.compile(r"^\s*(?:class|struct|interface)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)


def parse_native_module(file_path: Path, _project_root: Path) -> ParsedModule:
    source = file_path.read_text(encoding="utf-8", errors="ignore")
    parsed = ParsedModule(parse_mode="full")

    parsed.imports.extend(sorted(set(_IMPORT_RE.findall(source))))
    parsed.import_roots.update(item.split("/")[0] for item in parsed.imports if item)
    parsed.functions.extend(sorted(set(_FUNC_RE.findall(source))))

    for class_name in sorted(set(_CLASS_RE.findall(source))):
        parsed.classes.append(ClassDescriptor(name=class_name, methods=[], fields=[], decorators=[], bases=[]))

    return parsed
