from __future__ import annotations

import re
from pathlib import Path

from mana_analyzer.analysis.models import ClassDescriptor, ExportDescriptor
from mana_analyzer.services.parsers.base import ParsedModule

_IMPORT_RE = re.compile(r"(?:import\s+.*?from\s+|require\()\s*['\"]([^'\"]+)['\"]")
_EXPORT_RE = re.compile(r"^\s*export\s+(?:default\s+)?(?:class|function|const|let|var|interface|type)?\s*([A-Za-z_][A-Za-z0-9_]*)?", re.MULTILINE)
_FUNCTION_RE = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_CLASS_RE = re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_CONST_RE = re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)


def parse_js_ts_module(file_path: Path, project_root: Path) -> ParsedModule:
    source = file_path.read_text(encoding="utf-8", errors="ignore")
    module_path = str(file_path.relative_to(project_root))
    parsed = ParsedModule(parse_mode="full")

    for item in _IMPORT_RE.findall(source):
        parsed.imports.append(item)
        root_name = item.split("/")[0].split(".")[0]
        if root_name:
            parsed.import_roots.add(root_name)

    parsed.functions.extend(sorted(set(_FUNCTION_RE.findall(source))))

    for class_name in sorted(set(_CLASS_RE.findall(source))):
        parsed.classes.append(ClassDescriptor(name=class_name, methods=[], fields=[], decorators=[], bases=[]))
        parsed.exports.append(ExportDescriptor(source_module=module_path, symbol=class_name, mechanism="public-class"))

    parsed.constants.extend(sorted(set(_CONST_RE.findall(source))))

    for symbol in _EXPORT_RE.findall(source):
        if symbol:
            parsed.exports.append(ExportDescriptor(source_module=module_path, symbol=symbol, mechanism="public-export"))

    for fn_name in parsed.functions:
        if not fn_name.startswith("_"):
            parsed.exports.append(ExportDescriptor(source_module=module_path, symbol=fn_name, mechanism="public-function"))

    dedup = {(item.source_module, item.symbol, item.mechanism): item for item in parsed.exports}
    parsed.exports = sorted(dedup.values(), key=lambda item: (item.symbol, item.mechanism))

    return parsed
