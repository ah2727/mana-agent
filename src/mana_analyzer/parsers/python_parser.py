from __future__ import annotations

import ast
import logging
from pathlib import Path

from mana_analyzer.analysis.models import CodeSymbol

logger = logging.getLogger(__name__)


class _SymbolVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str, source: str) -> None:
        self.file_path = file_path
        self.source = source
        self.lines = source.splitlines()
        self.symbols: list[CodeSymbol] = []

    def _segment(self, start_line: int, end_line: int) -> str:
        start_idx = max(start_line - 1, 0)
        end_idx = max(end_line, start_idx)
        return "\n".join(self.lines[start_idx:end_idx])

    def _signature(self, node: ast.AST) -> str:
        text = ast.get_source_segment(self.source, node)
        if not text:
            return ""
        return text.splitlines()[0].strip()

    def _add_symbol(self, kind: str, name: str, signature: str, docstring: str, node: ast.AST) -> None:
        start_line = int(getattr(node, "lineno", 1))
        end_line = int(getattr(node, "end_lineno", start_line))
        self.symbols.append(
            CodeSymbol(
                kind=kind,
                name=name,
                signature=signature,
                docstring=docstring,
                file_path=self.file_path,
                start_line=start_line,
                end_line=end_line,
                source=self._segment(start_line, end_line),
            )
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._add_symbol(
            kind="function",
            name=node.name,
            signature=self._signature(node),
            docstring=ast.get_docstring(node) or "",
            node=node,
        )
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add_symbol(
            kind="async_function",
            name=node.name,
            signature=self._signature(node),
            docstring=ast.get_docstring(node) or "",
            node=node,
        )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases = [ast.unparse(base) for base in node.bases] if node.bases else []
        if bases:
            signature = f"class {node.name}({', '.join(bases)}):"
        else:
            signature = f"class {node.name}:"
        self._add_symbol(
            kind="class",
            name=node.name,
            signature=signature,
            docstring=ast.get_docstring(node) or "",
            node=node,
        )
        self.generic_visit(node)


class PythonParser:
    def parse_file(self, path: str | Path) -> list[CodeSymbol]:
        file_path = str(Path(path).resolve())
        logger.debug("Parsing Python file %s", file_path)
        source = Path(path).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=file_path)

        lines = source.splitlines()
        module_end = len(lines)
        module_symbol = CodeSymbol(
            kind="module",
            name=Path(path).name,
            signature=f"module {Path(path).name}",
            docstring=ast.get_docstring(tree) or "",
            file_path=file_path,
            start_line=1,
            end_line=module_end,
            source=source,
        )

        visitor = _SymbolVisitor(file_path=file_path, source=source)
        visitor.visit(tree)
        symbols = [module_symbol, *visitor.symbols]
        logger.debug("Parsed %d symbols from %s", len(symbols), file_path)
        return symbols
