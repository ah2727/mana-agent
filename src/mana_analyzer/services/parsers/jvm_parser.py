from __future__ import annotations

import re
from pathlib import Path

from mana_analyzer.analysis.models import ClassDescriptor, ExportDescriptor
from mana_analyzer.services.parsers.base import ParsedModule

_IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z0-9_.*]+)", re.MULTILINE)
_CLASS_RE = re.compile(r"^\s*(?:public\s+|private\s+|protected\s+)?(?:abstract\s+)?(?:class|interface|enum|data\s+class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_FUNCTION_RE = re.compile(r"^\s*(?:public|private|protected|internal)?\s*(?:static\s+)?[A-Za-z0-9_<>,?\[\]]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)


def parse_jvm_module(file_path: Path, project_root: Path) -> ParsedModule:
    source = file_path.read_text(encoding="utf-8", errors="ignore")
    module_path = str(file_path.relative_to(project_root))
    parsed = ParsedModule(parse_mode="full")

    parsed.imports.extend(sorted(set(_IMPORT_RE.findall(source))))
    parsed.import_roots.update(item.split(".")[0] for item in parsed.imports if item)

    parsed.functions.extend(sorted(set(_FUNCTION_RE.findall(source))))

    for class_name in sorted(set(_CLASS_RE.findall(source))):
        class_desc = ClassDescriptor(name=class_name, methods=[], fields=[], decorators=[], bases=[])
        parsed.classes.append(class_desc)
        parsed.exports.append(ExportDescriptor(source_module=module_path, symbol=class_name, mechanism="public-class"))

    for fn_name in parsed.functions:
        parsed.exports.append(ExportDescriptor(source_module=module_path, symbol=fn_name, mechanism="public-function"))

    return parsed
