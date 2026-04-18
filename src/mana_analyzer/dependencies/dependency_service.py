# mana_analyzer/dependencies/dependency_service.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Dict, List, Set

import ast


# ---------- Result Model ----------

@dataclass(frozen=True)
class DependencyAnalysisResult:
    files: List[Path]
    language_map: Dict[Path, str]
    source_map: Dict[Path, str]
    imports: Dict[Path, List[str]]


# ---------- Service ----------

class DependencyService:
    """
    Performs a single-pass repository scan:
      - Collects source files
      - Detects language
      - Reads source code
      - Extracts imports (Python only for now)
    """

    SUPPORTED_EXTENSIONS: Dict[str, str] = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".go": "go",
    }

    EXCLUDED_DIRS: Set[str] = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        "dist",
        "build",
    }

    def analyze(self, root: Path) -> DependencyAnalysisResult:
        """
        Analyzes the project at the given root path for dependencies.

        Args:
            root: The root path of the project to analyze.

        Returns:
            A DependencyAnalysisResult object containing the analysis findings.
        """
        root = root.resolve()

        files: List[Path] = []
        language_map: Dict[Path, str] = {}
        source_map: Dict[Path, str] = {}
        imports: Dict[Path, List[str]] = {}

        for path in self._walk_files(root):
            ext = path.suffix.lower()
            language = self.SUPPORTED_EXTENSIONS.get(ext)
            if not language:
                continue

            try:
                source = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:  # More specific exception for file reading errors
                continue
            except Exception:  # Catch any other unexpected exceptions
                continue

            files.append(path)
            language_map[path] = language
            source_map[path] = source
            imports[path] = self._extract_imports(path, source, language)

        return DependencyAnalysisResult(
            files=files,
            language_map=language_map,
            source_map=source_map,
            imports=imports,
        )

    # ---------- Helpers ----------

    def _walk_files(self, root: Path) -> Iterable[Path]:
        """
        Recursively walks through directories to yield supported files.

        Args:
            root: The root directory to start walking from.

        Yields:
            Paths to files within the root directory, excluding excluded directories.
        """
        for path in root.rglob("*"):
            if path.is_dir():
                continue
            # Check if any part of the path is in EXCLUDED_DIRS
            if any(part in self.EXCLUDED_DIRS for part in path.parts):
                continue
            yield path

    def _extract_imports(
        self,
        path: Path,
        source: str,
        language: str,
    ) -> List[str]:
        """
        Extracts import statements from the source code based on the language.

        Args:
            path: The path to the file being analyzed.
            source: The source code of the file.
            language: The detected language of the file.

        Returns:
            A list of imported module names.
        """
        if language == "python":
            return self._extract_python_imports(source)
        # Add support for other languages here as needed
        return []

    def _extract_python_imports(self, source: str) -> List[str]:
        """
        Extracts import statements from Python source code using the ast module.

        Args:
            source: The Python source code as a string.

        Returns:
            A sorted list of unique imported module names.
        """
        imports: Set[str] = set()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Silently ignore files with syntax errors
            return []
        except Exception:
            # Catch any other unexpected parsing errors
            return []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.add(name.name)
            elif isinstance(node, ast.ImportFrom):
                # Only add the module name if it's not None (e.g., for relative imports like from . import X)
                if node.module:
                    imports.add(node.module)

        return sorted(list(imports))
