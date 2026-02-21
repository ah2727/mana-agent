from __future__ import annotations

import ast
from pathlib import Path

from mana_analyzer.analysis.models import ClassDescriptor, ExportDescriptor
from mana_analyzer.services.parsers.base import ParsedModule


def _str_expr(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "<unknown>"


def parse_python_module(file_path: Path, project_root: Path) -> ParsedModule:
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file_path))
    module_path = str(file_path.relative_to(project_root))

    parsed = ParsedModule(parse_mode="full")

    for node in tree.body:
        if isinstance(node, ast.Import):
            for item in node.names:
                parsed.imports.append(item.name)
                parsed.import_roots.add(item.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = ",".join(item.name for item in node.names)
            parsed.imports.append(f"{module}:{names}")
            if module:
                parsed.import_roots.add(module.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parsed.functions.append(node.name)
            if not node.name.startswith("_"):
                parsed.exports.append(
                    ExportDescriptor(
                        source_module=module_path,
                        symbol=node.name,
                        mechanism="public-function",
                    )
                )
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call) and _str_expr(decorator.func) == "app.command":
                    parsed.commands.append(node.name)
                    parsed.exports.append(
                        ExportDescriptor(
                            source_module=module_path,
                            symbol=node.name,
                            mechanism="cli-command",
                        )
                    )
        elif isinstance(node, ast.ClassDef):
            methods: list[str] = []
            fields: list[str] = []
            decorators = [_str_expr(item) for item in node.decorator_list]
            bases = [_str_expr(item) for item in node.bases]
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(sub.name)
                elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                    fields.append(sub.target.id)
                elif isinstance(sub, ast.Assign):
                    for target in sub.targets:
                        if isinstance(target, ast.Name):
                            fields.append(target.id)
            class_desc = ClassDescriptor(
                name=node.name,
                methods=methods,
                fields=sorted(set(fields)),
                decorators=decorators,
                bases=bases,
            )
            parsed.classes.append(class_desc)
            parsed.data_structures.append(class_desc)
            if not node.name.startswith("_"):
                parsed.exports.append(
                    ExportDescriptor(
                        source_module=module_path,
                        symbol=node.name,
                        mechanism="public-class",
                    )
                )
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    parsed.constants.append(target.id)
                    if target.id == "__all__":
                        values: list[str] = []
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            for item in node.value.elts:
                                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                                    values.append(item.value)
                        for symbol in values:
                            parsed.exports.append(
                                ExportDescriptor(
                                    source_module=module_path,
                                    symbol=symbol,
                                    mechanism="__all__",
                                )
                            )

    return parsed
