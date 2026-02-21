from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path

from mana_analyzer.utils.io import EXCLUDED_DIRS, load_ignore_patterns

MANIFEST_FILENAMES = {
    "package.json",
    "pubspec.yaml",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "Pipfile",
    "setup.py",
    "nest-cli.json",
    "pom.xml",
    "Cargo.toml",
    "go.mod",
    "Gemfile",
    "composer.json",
}

MANIFEST_GLOBS = {
    "vite.config.*",
    "next.config.*",
    "nuxt.config.*",
    "build.gradle*",
    "settings.gradle*",
}

PACKAGE_MANAGER_FILES = {
    "uv.lock": "uv",
    "poetry.lock": "poetry",
    "Pipfile": "pipenv",
    "requirements.txt": "pip",
    "requirements-dev.txt": "pip",
    "pyproject.toml": "pip/setuptools",
    "package.json": "npm",
    "package-lock.json": "npm",
    "yarn.lock": "yarn",
    "pnpm-lock.yaml": "pnpm",
    "bun.lockb": "bun",
    "bun.lock": "bun",
    "pubspec.yaml": "pub",
    "Cargo.toml": "cargo",
    "go.mod": "go modules",
    "Gemfile": "bundler",
    "composer.json": "composer",
    "pom.xml": "maven",
    "build.gradle": "gradle",
    "build.gradle.kts": "gradle",
    "settings.gradle": "gradle",
    "settings.gradle.kts": "gradle",
}


@dataclass(slots=True)
class SubprojectDescriptor:
    root_path: Path
    manifest_paths: list[Path]
    package_managers: list[str]
    framework_hints: list[str]


def _matches_ignore(relative_path: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        normalized = pattern.lstrip("./")
        if fnmatch.fnmatch(relative_path, normalized):
            return True
        if relative_path.startswith(normalized.rstrip("/") + "/"):
            return True
    return False


def _is_manifest(name: str) -> bool:
    if name in MANIFEST_FILENAMES:
        return True
    return any(fnmatch.fnmatch(name, pattern) for pattern in MANIFEST_GLOBS)


def _framework_hints(paths: list[Path]) -> list[str]:
    names = {item.name for item in paths}
    hints: set[str] = set()
    if "pubspec.yaml" in names:
        hints.add("Flutter")
    if "nest-cli.json" in names:
        hints.add("NestJS")
    if "vite.config.ts" in names or "vite.config.js" in names or "vite.config.mjs" in names:
        hints.add("Vite")
    if any(name.startswith("next.config.") for name in names):
        hints.add("Next.js")
    if any(name.startswith("nuxt.config.") for name in names):
        hints.add("Nuxt")
    return sorted(hints)


def _package_managers(subproject_root: Path, manifests: list[Path]) -> list[str]:
    managers = {PACKAGE_MANAGER_FILES[item.name] for item in manifests if item.name in PACKAGE_MANAGER_FILES}
    for filename, manager in PACKAGE_MANAGER_FILES.items():
        if (subproject_root / filename).exists():
            managers.add(manager)
    return sorted(managers)


def discover_subprojects(root: str | Path) -> list[SubprojectDescriptor]:
    root_path = Path(root).resolve()
    if root_path.is_file():
        root_path = root_path.parent

    ignore_patterns = load_ignore_patterns(root_path)
    manifests_by_dir: dict[Path, list[Path]] = {}

    for path in root_path.rglob("*"):
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        if not path.is_file() or not _is_manifest(path.name):
            continue
        relative = str(path.relative_to(root_path))
        if _matches_ignore(relative, ignore_patterns):
            continue
        manifests_by_dir.setdefault(path.parent, []).append(path)

    subprojects: list[SubprojectDescriptor] = []
    for sub_root, manifest_paths in manifests_by_dir.items():
        manifests_sorted = sorted(manifest_paths)
        subprojects.append(
            SubprojectDescriptor(
                root_path=sub_root,
                manifest_paths=manifests_sorted,
                package_managers=_package_managers(sub_root, manifests_sorted),
                framework_hints=_framework_hints(manifests_sorted),
            )
        )

    return sorted(subprojects, key=lambda item: str(item.root_path.relative_to(root_path)))
