from __future__ import annotations

import re
from pathlib import Path

from mana_analyzer.analysis.models import ClassDescriptor, ExportDescriptor
from mana_analyzer.services.parsers.base import ParsedModule

_IMPORT_RE = re.compile(r"^\s*import\s+['\"]([^'\"]+)['\"]", re.MULTILINE)
_CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_FUNCTION_RE = re.compile(r"^\s*(?:[A-Za-z_<>,?\[\]\s]+)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{]*\)\s*\{", re.MULTILINE)
_CONST_RE = re.compile(r"^\s*(?:final|const|var)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)


def parse_dart_module(file_path: Path, project_root: Path) -> ParsedModule:
    source = file_path.read_text(encoding="utf-8", errors="ignore")
    module_path = str(file_path.relative_to(project_root))
    parsed = ParsedModule(parse_mode="full")

    for item in _IMPORT_RE.findall(source):
        parsed.imports.append(item)
        root_name = item.split(":")[0].split("/")[0]
        if root_name:
            parsed.import_roots.add(root_name)

    parsed.functions.extend(sorted(set(name for name in _FUNCTION_RE.findall(source) if name and name[0].islower())))
    parsed.constants.extend(sorted(set(_CONST_RE.findall(source))))

    for class_name in sorted(set(_CLASS_RE.findall(source))):
        class_desc = ClassDescriptor(name=class_name, methods=[], fields=[], decorators=[], bases=[])
        parsed.classes.append(class_desc)
        parsed.exports.append(ExportDescriptor(source_module=module_path, symbol=class_name, mechanism="public-class"))

    for fn_name in parsed.functions:
        parsed.exports.append(ExportDescriptor(source_module=module_path, symbol=fn_name, mechanism="public-function"))

    return parsed
