from __future__ import annotations

import re
from pathlib import Path

from mana_analyzer.analysis.models import ClassDescriptor, ExportDescriptor
from mana_analyzer.services.parsers.base import ParsedModule

_IMPORT_RE = re.compile(r"^\s*#\s*include\s+[<\"]([^>\"]+)[>\"]", re.MULTILINE)
_FUNC_RE = re.compile(r"^\s*(?:[A-Za-z_][A-Za-z0-9_\s:*<>]+)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{", re.MULTILINE)
_CLASS_RE = re.compile(r"^\s*(?:class|struct|interface)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)

_GO_IMPORT_SINGLE_RE = re.compile(r'^\s*import\s+(?:[A-Za-z0-9_\.]+\s+)?"([^\"]+)"', re.MULTILINE)
_GO_IMPORT_BLOCK_RE = re.compile(r"import\s*\((.*?)\)", re.DOTALL)
_GO_FUNC_RE = re.compile(r"^\s*func\s*(?:\([^)]+\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
_GO_METHOD_RE = re.compile(
    r"^\s*func\s*\(\s*[^)]*\*?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(",
    re.MULTILINE,
)
_GO_TYPE_RE = re.compile(r"^\s*type\s+([A-Za-z_][A-Za-z0-9_]*)\s+(struct|interface|[A-Za-z_\[])", re.MULTILINE)
_GO_CONST_BLOCK_RE = re.compile(r"(?ms)^\s*const\s*\((.*?)^\s*\)")
_GO_CONST_LINE_RE = re.compile(r"^\s*const\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_GO_VAR_BLOCK_RE = re.compile(r"(?ms)^\s*var\s*\((.*?)^\s*\)")
_GO_VAR_LINE_RE = re.compile(r"^\s*var\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_GO_COMMAND_RE = re.compile(r'\bUse\s*:\s*"([^\"]+)"')


def _is_public_go_symbol(name: str) -> bool:
    return bool(name) and name[0].isupper()


def _parse_go_module(file_path: Path, project_root: Path, source: str) -> ParsedModule:
    module_path = str(file_path.relative_to(project_root))
    parsed = ParsedModule(parse_mode="full")

    imports: set[str] = set(_GO_IMPORT_SINGLE_RE.findall(source))
    for block in _GO_IMPORT_BLOCK_RE.findall(source):
        imports.update(re.findall(r'"([^\"]+)"', block))

    parsed.imports.extend(sorted(imports))
    parsed.import_roots.update(item.split("/")[0] for item in parsed.imports if item)

    parsed.functions.extend(sorted(set(_GO_FUNC_RE.findall(source))))

    methods_by_type: dict[str, list[str]] = {}
    for receiver_type, method_name in _GO_METHOD_RE.findall(source):
        methods_by_type.setdefault(receiver_type, []).append(method_name)

    for type_name, kind_token in sorted(set(_GO_TYPE_RE.findall(source))):
        methods = sorted(set(methods_by_type.get(type_name, [])))
        fields: list[str] = []
        if kind_token == "struct":
            struct_match = re.search(
                rf"(?ms)^\s*type\s+{re.escape(type_name)}\s+struct\s*\{{(.*?)^\s*\}}",
                source,
            )
            if struct_match:
                for raw_line in struct_match.group(1).splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("//") or line.startswith("*"):
                        continue
                    field_name = line.split()[0]
                    if field_name and field_name != "}" and field_name != "{" and field_name != "//":
                        fields.append(field_name)

        class_desc = ClassDescriptor(
            name=type_name,
            methods=methods,
            fields=sorted(set(fields)),
            decorators=[],
            bases=[],
        )
        parsed.classes.append(class_desc)
        parsed.data_structures.append(class_desc)

        if _is_public_go_symbol(type_name):
            parsed.exports.append(
                ExportDescriptor(source_module=module_path, symbol=type_name, mechanism="public-type")
            )

    for func_name in parsed.functions:
        if _is_public_go_symbol(func_name):
            parsed.exports.append(
                ExportDescriptor(source_module=module_path, symbol=func_name, mechanism="public-function")
            )

    consts: set[str] = set(_GO_CONST_LINE_RE.findall(source))
    for block in _GO_CONST_BLOCK_RE.findall(source):
        for line in block.splitlines():
            token = line.strip().split()
            if token and token[0].isidentifier():
                consts.add(token[0])
    parsed.constants.extend(sorted(consts))
    for const_name in sorted(consts):
        if _is_public_go_symbol(const_name):
            parsed.exports.append(
                ExportDescriptor(source_module=module_path, symbol=const_name, mechanism="public-const")
            )

    vars_: set[str] = set(_GO_VAR_LINE_RE.findall(source))
    for block in _GO_VAR_BLOCK_RE.findall(source):
        for line in block.splitlines():
            token = line.strip().split()
            if token and token[0].isidentifier():
                vars_.add(token[0])
    for var_name in sorted(vars_):
        if _is_public_go_symbol(var_name):
            parsed.exports.append(
                ExportDescriptor(source_module=module_path, symbol=var_name, mechanism="public-var")
            )

    parsed.commands.extend(sorted(set(_GO_COMMAND_RE.findall(source))))

    dedup = {(item.source_module, item.symbol, item.mechanism): item for item in parsed.exports}
    parsed.exports = sorted(dedup.values(), key=lambda item: (item.symbol, item.mechanism))

    return parsed


def parse_native_module(file_path: Path, project_root: Path) -> ParsedModule:
    source = file_path.read_text(encoding="utf-8", errors="ignore")

    if file_path.suffix.lower() == ".go":
        return _parse_go_module(file_path, project_root, source)

    parsed = ParsedModule(parse_mode="full")

    parsed.imports.extend(sorted(set(_IMPORT_RE.findall(source))))
    parsed.import_roots.update(item.split("/")[0] for item in parsed.imports if item)
    parsed.functions.extend(sorted(set(_FUNC_RE.findall(source))))

    for class_name in sorted(set(_CLASS_RE.findall(source))):
        parsed.classes.append(ClassDescriptor(name=class_name, methods=[], fields=[], decorators=[], bases=[]))

    return parsed
