from __future__ import annotations

import hashlib
import json
import fnmatch
from pathlib import Path
from typing import Iterable

EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "node_modules",
    "build",
    "dist",
    ".mana",
    ".mana_index",
    ".next",
    "coverage",
    ".dart_tool",
    "Pods",
    "target",
    "vendor",
    "out",
}

SOURCE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",
    ".dart",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".swift",
    ".m",
    ".mm",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".php",
    ".rb",
    ".scala",
    ".sh",
    ".bash",
    ".zsh",
    ".sql",
    ".html",
    ".css",
    ".scss",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".md",
}

LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".dart": "dart",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".swift": "swift",
    ".m": "objective-c",
    ".mm": "objective-c++",
    ".c": "c",
    ".cc": "c++",
    ".cpp": "c++",
    ".h": "c/c++ header",
    ".hpp": "c++ header",
    ".cs": "c#",
    ".php": "php",
    ".rb": "ruby",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
}

IGNORE_FILES = (".gitignore", ".aiignore", ".aiexclude")


def load_ignore_patterns(root: str | Path) -> list[str]:
    root_path = Path(root).resolve()
    patterns: list[str] = []
    for name in IGNORE_FILES:
        target = root_path / name
        if not target.exists():
            continue
        for line in target.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            patterns.append(item)
    return patterns


def _matches_ignore(relative_path: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        normalized = pattern.lstrip("./")
        if fnmatch.fnmatch(relative_path, normalized):
            return True
        if relative_path.startswith(normalized.rstrip("/") + "/"):
            return True
    return False


def iter_source_files(root: str | Path, extensions: set[str] | None = None) -> list[Path]:
    root_path = Path(root).resolve()
    allowed = extensions or SOURCE_EXTENSIONS
    
    if root_path.is_file():
        return [root_path] if root_path.suffix.lower() in allowed else []

    ignore_patterns = load_ignore_patterns(root_path)
    found_files: list[Path] = []

    # Use os.walk for better performance and easy directory skipping
    import os
    for res_root, dirs, files in os.walk(root_path):
        # 1. Prune excluded directories in-place (stops walk from entering them)
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith('.')]
        
        rel_dir = os.path.relpath(res_root, root_path)
        
        for file in files:
            file_path = Path(res_root) / file
            
            # 2. Check Extension (Case Insensitive)
            if file_path.suffix.lower() not in allowed:
                continue
                
            # 3. Check Ignore Patterns (relative to root)
            rel_file_path = os.path.join(rel_dir, file) if rel_dir != "." else file
            if _matches_ignore(rel_file_path, ignore_patterns):
                continue
                
            found_files.append(file_path)

    return sorted(found_files)


def iter_python_files(root: str | Path) -> list[Path]:
    return iter_source_files(root, extensions={".py"})


def language_for_path(path: str | Path) -> str:
    return LANGUAGE_BY_EXTENSION.get(Path(path).suffix.lower(), "unknown")


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_text(read_text(path))


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: str | Path) -> dict:
    target = Path(path)
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
