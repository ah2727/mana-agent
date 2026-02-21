from __future__ import annotations

import re
from pathlib import Path

from mana_analyzer.analysis.models import CodeSymbol
from mana_analyzer.parsers.python_parser import PythonParser

_JS_TS_FUNCTION_RE = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)
_JS_TS_CLASS_RE = re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_JS_TS_CONST_RE = re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)

_DART_FUNCTION_RE = re.compile(
    r"^\s*(?:[A-Za-z_<>,?\[\]\s]+)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{]*\)\s*\{",
    re.MULTILINE,
)
_DART_CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_DART_CONST_RE = re.compile(r"^\s*(?:final|const|var)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)

_JVM_FUNCTION_RE = re.compile(
    r"^\s*(?:public|private|protected|internal)?\s*(?:static\s+)?[A-Za-z0-9_<>,?\[\]]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
    re.MULTILINE,
)
_JVM_CLASS_RE = re.compile(
    r"^\s*(?:public\s+|private\s+|protected\s+)?(?:abstract\s+)?(?:class|interface|enum|data\s+class)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)

_NATIVE_FUNCTION_RE = re.compile(
    r"^\s*(?:[A-Za-z_][A-Za-z0-9_\s:*<>]+)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{",
    re.MULTILINE,
)
_NATIVE_CLASS_RE = re.compile(r"^\s*(?:class|struct|interface)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)


class MultiLanguageParser:
    def __init__(self) -> None:
        self.python_parser = PythonParser()

    @staticmethod
    def _line_number(source: str, offset: int) -> int:
        return source.count("\n", 0, offset) + 1

    @staticmethod
    def _line_segment(lines: list[str], line_no: int) -> str:
        idx = max(line_no - 1, 0)
        if idx >= len(lines):
            return ""
        return lines[idx]

    @staticmethod
    def _module_symbol(path: Path, source: str) -> CodeSymbol:
        lines = source.splitlines()
        end_line = len(lines) if lines else 1
        return CodeSymbol(
            kind="module",
            name=path.name,
            signature=f"module {path.name}",
            docstring="",
            file_path=str(path.resolve()),
            start_line=1,
            end_line=end_line,
            source=source,
        )

    def _extract_symbols(
        self,
        path: Path,
        source: str,
        function_re: re.Pattern[str] | None,
        class_re: re.Pattern[str] | None,
        const_re: re.Pattern[str] | None,
    ) -> list[CodeSymbol]:
        lines = source.splitlines()
        file_path = str(path.resolve())
        output: list[CodeSymbol] = [self._module_symbol(path, source)]
        seen: set[tuple[str, str, int]] = set()

        def add(kind: str, pattern: re.Pattern[str] | None) -> None:
            if pattern is None:
                return
            for match in pattern.finditer(source):
                name = match.group(1)
                if not name:
                    continue
                line_no = self._line_number(source, match.start())
                key = (kind, name, line_no)
                if key in seen:
                    continue
                seen.add(key)
                signature = self._line_segment(lines, line_no).strip()
                output.append(
                    CodeSymbol(
                        kind=kind,
                        name=name,
                        signature=signature,
                        docstring="",
                        file_path=file_path,
                        start_line=line_no,
                        end_line=line_no,
                        source=self._line_segment(lines, line_no),
                    )
                )

        add("function", function_re)
        add("class", class_re)
        add("constant", const_re)
        return output

    def parse_file(self, path: str | Path) -> list[CodeSymbol]:
        target = Path(path)
        suffix = target.suffix.lower()

        if suffix == ".py":
            return self.python_parser.parse_file(target)

        source = target.read_text(encoding="utf-8", errors="ignore")

        if suffix in {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}:
            return self._extract_symbols(target, source, _JS_TS_FUNCTION_RE, _JS_TS_CLASS_RE, _JS_TS_CONST_RE)
        if suffix == ".dart":
            return self._extract_symbols(target, source, _DART_FUNCTION_RE, _DART_CLASS_RE, _DART_CONST_RE)
        if suffix in {".java", ".kt"}:
            return self._extract_symbols(target, source, _JVM_FUNCTION_RE, _JVM_CLASS_RE, None)
        if suffix in {".swift", ".m", ".mm", ".c", ".cc", ".cpp", ".h", ".hpp", ".cs", ".scala", ".rs", ".go"}:
            return self._extract_symbols(target, source, _NATIVE_FUNCTION_RE, _NATIVE_CLASS_RE, None)

        return [self._module_symbol(target, source)]
