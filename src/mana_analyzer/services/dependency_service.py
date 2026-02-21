from __future__ import annotations

import ast
import json
import re
import tomllib
from pathlib import Path

from mana_analyzer.analysis.models import DependencyEdge, DependencyGraphReport
from mana_analyzer.utils.io import iter_source_files, language_for_path
from mana_analyzer.utils.project_discovery import discover_subprojects

PYTHON_EXTENSIONS = {".py"}
JS_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}

FRAMEWORK_SIGNALS = {
    "django": "Django",
    "flask": "Flask",
    "fastapi": "FastAPI",
    "typer": "Typer",
    "click": "Click",
    "react": "React",
    "vite": "Vite",
    "next": "Next.js",
    "nextjs": "Next.js",
    "vue": "Vue",
    "nuxt": "Nuxt",
    "svelte": "Svelte",
    "express": "Express",
    "nestjs": "NestJS",
    "langchain": "LangChain",
}


def _normalize_dep_name(raw: str) -> str:
    item = raw.strip().lower()
    if not item:
        return ""
    item = re.split(r"[<>=!~\[]", item)[0]
    return item.strip()


def _looks_external(name: str) -> bool:
    return bool(name) and not name.startswith(".")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class DependencyService:
    @staticmethod
    def _detect_languages(files: list[Path]) -> list[str]:
        langs = {language_for_path(item) for item in files}
        return sorted(item for item in langs if item and item != "unknown")

    @staticmethod
    def _parse_pyproject(path: Path) -> tuple[list[str], list[str]]:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        project = payload.get("project", {})
        runtime = [_normalize_dep_name(item) for item in project.get("dependencies", [])]
        optional = project.get("optional-dependencies", {})
        dev = [_normalize_dep_name(item) for item in optional.get("dev", [])]

        poetry = payload.get("tool", {}).get("poetry", {})
        poetry_runtime = [_normalize_dep_name(item) for item in poetry.get("dependencies", {}).keys()]
        poetry_dev = [_normalize_dep_name(item) for item in poetry.get("group", {}).get("dev", {}).get("dependencies", {}).keys()]
        runtime.extend(poetry_runtime)
        dev.extend(poetry_dev)

        return sorted({item for item in runtime if item and item != "python"}), sorted(
            {item for item in dev if item and item != "python"}
        )

    @staticmethod
    def _parse_requirements(path: Path) -> list[str]:
        deps: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if not item or item.startswith("#") or item.startswith("-"):
                continue
            deps.append(_normalize_dep_name(item))
        return sorted({item for item in deps if item})

    @staticmethod
    def _parse_pipfile(path: Path) -> tuple[list[str], list[str]]:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        runtime = [_normalize_dep_name(name) for name in payload.get("packages", {}).keys()]
        dev = [_normalize_dep_name(name) for name in payload.get("dev-packages", {}).keys()]
        return sorted({item for item in runtime if item}), sorted({item for item in dev if item})

    @staticmethod
    def _parse_package_json(path: Path) -> tuple[list[str], list[str]]:
        payload = _read_json(path)
        runtime = [_normalize_dep_name(name) for name in payload.get("dependencies", {}).keys()]
        dev = [_normalize_dep_name(name) for name in payload.get("devDependencies", {}).keys()]
        return sorted({item for item in runtime if item}), sorted({item for item in dev if item})

    @staticmethod
    def _parse_setup_py(path: Path) -> list[str]:
        matches = re.findall(r"['\"]([a-zA-Z0-9_.\-]+[<>=!~]?.*?)['\"]", path.read_text(encoding="utf-8"))
        return sorted({_normalize_dep_name(item) for item in matches if _normalize_dep_name(item)})

    @staticmethod
    def _parse_pubspec(path: Path) -> tuple[list[str], list[str]]:
        runtime: set[str] = set()
        dev: set[str] = set()
        section = ""
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped in {"dependencies:", "dev_dependencies:"}:
                section = stripped[:-1]
                continue
            if section and ":" in stripped and not stripped.startswith("-"):
                name = _normalize_dep_name(stripped.split(":", 1)[0])
                if not name:
                    continue
                if section == "dependencies":
                    runtime.add(name)
                elif section == "dev_dependencies":
                    dev.add(name)
        return sorted(runtime), sorted(dev)

    @staticmethod
    def _parse_composer(path: Path) -> tuple[list[str], list[str]]:
        payload = _read_json(path)
        runtime = [_normalize_dep_name(name) for name in payload.get("require", {}).keys()]
        dev = [_normalize_dep_name(name) for name in payload.get("require-dev", {}).keys()]
        return sorted({item for item in runtime if item}), sorted({item for item in dev if item})

    @staticmethod
    def _parse_gemfile(path: Path) -> tuple[list[str], list[str]]:
        runtime: set[str] = set()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped.startswith("gem "):
                continue
            match = re.search(r"gem\s+['\"]([^'\"]+)['\"]", stripped)
            if match:
                runtime.add(_normalize_dep_name(match.group(1)))
        return sorted(runtime), []

    @staticmethod
    def _parse_go_mod(path: Path) -> tuple[list[str], list[str]]:
        runtime: set[str] = set()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if stripped.startswith("require "):
                parts = stripped.split()
                if len(parts) >= 2:
                    runtime.add(_normalize_dep_name(parts[1]))
            elif stripped and not stripped.startswith(("module ", "go ", "(", ")", "//")):
                parts = stripped.split()
                if len(parts) >= 2 and "." in parts[0]:
                    runtime.add(_normalize_dep_name(parts[0]))
        return sorted(runtime), []

    @staticmethod
    def _parse_cargo(path: Path) -> tuple[list[str], list[str]]:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        runtime = [_normalize_dep_name(name) for name in payload.get("dependencies", {}).keys()]
        dev = [_normalize_dep_name(name) for name in payload.get("dev-dependencies", {}).keys()]
        return sorted({item for item in runtime if item}), sorted({item for item in dev if item})

    @staticmethod
    def _module_name(root: Path, path: Path) -> str:
        relative = path.relative_to(root)
        parts = list(relative.parts)
        if parts and parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
        if parts and parts[-1] in {"__init__", "index"}:
            parts = parts[:-1]
        if not parts:
            return str(relative)
        return ".".join(parts)

    @staticmethod
    def _extract_python_imports(path: Path) -> list[str]:
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            return []
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        return imports

    @staticmethod
    def _extract_js_imports(path: Path) -> list[str]:
        source = path.read_text(encoding="utf-8", errors="ignore")
        return re.findall(r"(?:from|require\()\s*['\"]([^'\"]+)['\"]", source)

    def _manifest_deps(self, manifest: Path) -> tuple[list[str], list[str]]:
        if manifest.name == "pyproject.toml":
            return self._parse_pyproject(manifest)
        if manifest.name.startswith("requirements"):
            return self._parse_requirements(manifest), []
        if manifest.name == "Pipfile":
            return self._parse_pipfile(manifest)
        if manifest.name == "package.json":
            return self._parse_package_json(manifest)
        if manifest.name == "setup.py":
            return self._parse_setup_py(manifest), []
        if manifest.name == "pubspec.yaml":
            return self._parse_pubspec(manifest)
        if manifest.name == "composer.json":
            return self._parse_composer(manifest)
        if manifest.name == "Gemfile":
            return self._parse_gemfile(manifest)
        if manifest.name == "go.mod":
            return self._parse_go_mod(manifest)
        if manifest.name == "Cargo.toml":
            return self._parse_cargo(manifest)
        return [], []

    @staticmethod
    def _match_frameworks(deps: set[str], manifest_names: set[str], subproject_hints: set[str]) -> list[str]:
        frameworks = {label for name, label in FRAMEWORK_SIGNALS.items() if name in deps}

        if any(dep.startswith("@nestjs/") for dep in deps) or "nest-cli.json" in manifest_names:
            frameworks.add("NestJS")
        if "react" in deps:
            frameworks.add("React")
        if "vite" in deps or any(name.startswith("vite.config.") for name in manifest_names):
            frameworks.add("Vite")
        if "pubspec.yaml" in manifest_names and ("flutter" in deps or "cupertino_icons" in deps):
            frameworks.add("Flutter")
        if any(name.startswith("next.config.") for name in manifest_names):
            frameworks.add("Next.js")
        if any(name.startswith("nuxt.config.") for name in manifest_names):
            frameworks.add("Nuxt")

        frameworks.update(subproject_hints)
        return sorted(frameworks)

    def analyze(self, target_path: str | Path) -> DependencyGraphReport:
        root = Path(target_path).resolve()
        if root.is_file():
            root = root.parent

        files = iter_source_files(root)
        subprojects = discover_subprojects(root)

        runtime_deps: set[str] = set()
        dev_deps: set[str] = set()
        manifests: set[Path] = set()
        package_managers: set[str] = set()
        subproject_hints: set[str] = set()
        manifest_names: set[str] = set()

        for subproject in subprojects:
            package_managers.update(subproject.package_managers)
            subproject_hints.update(subproject.framework_hints)
            for manifest in subproject.manifest_paths:
                manifests.add(manifest)
                manifest_names.add(manifest.name)
                runtime, dev = self._manifest_deps(manifest)
                runtime_deps.update(runtime)
                dev_deps.update(dev)

        module_edges: list[DependencyEdge] = []
        dependency_edges: list[DependencyEdge] = []

        known_modules: set[str] = set()
        known_roots: set[str] = set()
        file_to_module: dict[Path, str] = {}
        for file_path in files:
            module_name = self._module_name(root, file_path)
            known_modules.add(module_name)
            parts = module_name.split(".")
            if parts:
                known_roots.add(parts[0])
            if len(parts) > 1:
                known_roots.add(parts[-1])
            file_to_module[file_path] = module_name

        observed_import_roots: set[str] = set()
        for file_path in files:
            module_name = file_to_module[file_path]
            imports: list[str] = []
            if file_path.suffix in PYTHON_EXTENSIONS:
                imports = self._extract_python_imports(file_path)
            elif file_path.suffix in JS_EXTENSIONS:
                imports = self._extract_js_imports(file_path)

            for item in imports:
                item = item.strip()
                if not item:
                    continue
                root_name = item.split("/")[0].split(".")[0]
                observed_import_roots.add(root_name)
                is_internal = (
                    item in known_modules
                    or root_name in known_modules
                    or root_name in known_roots
                    or any(module.endswith(item) for module in known_modules)
                )
                if is_internal:
                    dependency_target = (
                        item if item in known_modules else next((module for module in known_modules if module.endswith(item)), root_name)
                    )
                    module_edges.append(
                        DependencyEdge(
                            source=module_name,
                            target=dependency_target,
                            kind="module-import",
                            file_path=str(file_path),
                        )
                    )
                elif _looks_external(root_name):
                    dependency_edges.append(
                        DependencyEdge(
                            source=module_name,
                            target=root_name,
                            kind="external-import",
                            file_path=str(file_path),
                        )
                    )

        all_deps = runtime_deps | dev_deps | observed_import_roots
        frameworks = self._match_frameworks(all_deps, manifest_names, subproject_hints)
        technologies = sorted(set(frameworks) | package_managers)

        return DependencyGraphReport(
            project_root=str(root),
            package_managers=sorted(package_managers),
            frameworks=frameworks,
            technologies=technologies,
            runtime_dependencies=sorted(runtime_deps),
            dev_dependencies=sorted(dev_deps),
            module_edges=sorted(module_edges, key=lambda item: (item.source, item.target, item.kind)),
            dependency_edges=sorted(dependency_edges, key=lambda item: (item.source, item.target, item.kind)),
            manifests=sorted(str(item.relative_to(root)) for item in manifests),
            languages=self._detect_languages(files),
        )
