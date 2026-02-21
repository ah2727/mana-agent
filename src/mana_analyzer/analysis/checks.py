from __future__ import annotations

import ast
import logging
from pathlib import Path

from mana_analyzer.analysis.models import Finding

logger = logging.getLogger(__name__)


def _is_public_name(name: str) -> bool:
    return not name.startswith("_")


class _UsedNameVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        self.names.add(node.id)
        self.generic_visit(node)


class _MaxNestingVisitor(ast.NodeVisitor):
    CONTROL_NODES = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AsyncFor, ast.AsyncWith)

    def __init__(self) -> None:
        self.max_depth = 0
        self._depth = 0
        self.deep_nodes: list[ast.AST] = []

    def generic_visit(self, node: ast.AST) -> None:
        is_control = isinstance(node, self.CONTROL_NODES)
        if is_control:
            self._depth += 1
            self.max_depth = max(self.max_depth, self._depth)
            if self._depth > 3:
                self.deep_nodes.append(node)

        super().generic_visit(node)

        if is_control:
            self._depth -= 1


class PythonStaticAnalyzer:
    def analyze_file(self, path: str | Path) -> list[Finding]:
        file_path = str(Path(path).resolve())
        logger.debug("Running static checks for %s", file_path)
        source = Path(path).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=file_path)

        findings: list[Finding] = []
        findings.extend(self._check_wildcard_import(tree, file_path))
        findings.extend(self._check_unused_imports(tree, file_path))
        findings.extend(self._check_missing_docstrings(tree, file_path))
        findings.extend(self._check_deep_nesting(tree, file_path))
        logger.debug("Static checks generated %d findings for %s", len(findings), file_path)
        return findings

    def _check_wildcard_import(self, tree: ast.AST, file_path: str) -> list[Finding]:
        findings: list[Finding] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if any(alias.name == "*" for alias in node.names):
                    findings.append(
                        Finding(
                            rule_id="wildcard-import",
                            severity="error",
                            message="Wildcard import detected.",
                            file_path=file_path,
                            line=int(getattr(node, "lineno", 1)),
                            column=int(getattr(node, "col_offset", 0)),
                        )
                    )
        return findings

    def _check_unused_imports(self, tree: ast.AST, file_path: str) -> list[Finding]:
        imports: list[tuple[str, ast.AST]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    ref_name = alias.asname or alias.name.split(".")[0]
                    imports.append((ref_name, node))
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    ref_name = alias.asname or alias.name
                    imports.append((ref_name, node))

        used = _UsedNameVisitor()
        used.visit(tree)

        findings: list[Finding] = []
        for ref_name, node in imports:
            if ref_name not in used.names:
                findings.append(
                    Finding(
                        rule_id="unused-imports",
                        severity="error",
                        message=f"Imported name '{ref_name}' is unused.",
                        file_path=file_path,
                        line=int(getattr(node, "lineno", 1)),
                        column=int(getattr(node, "col_offset", 0)),
                    )
                )
        return findings

    def _check_missing_docstrings(self, tree: ast.Module, file_path: str) -> list[Finding]:
        findings: list[Finding] = []
        if ast.get_docstring(tree) is None:
            findings.append(
                Finding(
                    rule_id="missing-docstring",
                    severity="warning",
                    message="Module is missing a docstring.",
                    file_path=file_path,
                    line=1,
                    column=0,
                )
            )

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if _is_public_name(node.name) and ast.get_docstring(node) is None:
                    findings.append(
                        Finding(
                            rule_id="missing-docstring",
                            severity="warning",
                            message=f"{node.__class__.__name__} '{node.name}' is missing a docstring.",
                            file_path=file_path,
                            line=int(getattr(node, "lineno", 1)),
                            column=int(getattr(node, "col_offset", 0)),
                        )
                    )
        return findings

    def _check_deep_nesting(self, tree: ast.AST, file_path: str) -> list[Finding]:
        visitor = _MaxNestingVisitor()
        visitor.visit(tree)

        findings: list[Finding] = []
        for node in visitor.deep_nodes:
            findings.append(
                Finding(
                    rule_id="deep-nesting",
                    severity="warning",
                    message="Nesting depth exceeds 3 levels.",
                    file_path=file_path,
                    line=int(getattr(node, "lineno", 1)),
                    column=int(getattr(node, "col_offset", 0)),
                )
            )
        return findings
