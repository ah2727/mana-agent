from __future__ import annotations

import ast
import json
import re
import tomllib
from pathlib import Path

from mana_analyzer.analysis.models import DependencyEdge, DependencyGraphReport
from mana_analyzer.utils.io import iter_source_files, language_for_path
from mana_analyzer.utils.project_discovery import discover_subprojects
from mana_analyzer.models import DependencyPackageRef

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

_EXACT_SEMVER_RE = re.compile(r"^\s*v?(\d+\.\d+\.\d+(?:[.\-+][0-9A-Za-z.\-+]+)?)\s*$")
_PY_EQ_RE = re.compile(r"^\s*==\s*v?(\d+\.\d+\.\d+(?:[.\-+][0-9A-Za-z.\-+]+)?)\s*$")


def _detect_exact_version(raw: str) -> str | None:
    if not raw:
        return None
    s = raw.strip()
    m = _PY_EQ_RE.match(s)
    if m:
        return m.group(1)
    # common exact pins in manifests: "1.2.3" or "v1.2.3"
    m = _EXACT_SEMVER_RE.match(s)
    if m:
        return m.group(1)
    return None

_REQ_NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)")


def _parse_python_req_line(raw: str) -> tuple[str, str]:
    """
    Input example: "requests>=2.0,<3" or "fastapi==0.110.0; python_version>='3.10'"
    Returns (name, version_spec_raw) where version_spec_raw is everything after name up to ';' (trimmed).
    If no spec part found, version_spec_raw="".
    """
    s = raw.strip()
    if not s:
        return "", ""
    # strip environment markers for v1
    left = s.split(";", 1)[0].strip()
    m = _REQ_NAME_RE.match(left)
    if not m:
        return "", ""
    name = m.group(1)
    spec = left[len(name):].strip()
    return _normalize_dep_name(name), spec

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

    @staticmethod
    def _inventory_pyproject(path: Path) -> tuple[list[DependencyPackageRef], list[DependencyPackageRef]]:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        runtime_items: list[DependencyPackageRef] = []
        dev_items: list[DependencyPackageRef] = []

        # PEP 621: project.dependencies = ["name>=x", ...]
        project = payload.get("project", {})
        for raw in project.get("dependencies", []) or []:
            name, spec = _parse_python_req_line(str(raw))
            if not name or name == "python":
                continue
            runtime_items.append(
                DependencyPackageRef(
                    name=name,
                    ecosystem="PyPI",
                    scope="runtime",
                    manifest_path=str(path),
                    package_manager="pip",
                    version_spec_raw=spec,
                    exact_version=_detect_exact_version(spec),
                )
            )

        # optional-dependencies.dev = ["name==x", ...]
        optional = project.get("optional-dependencies", {}) or {}
        for raw in (optional.get("dev") or []):
            name, spec = _parse_python_req_line(str(raw))
            if not name or name == "python":
                continue
            dev_items.append(
                DependencyPackageRef(
                    name=name,
                    ecosystem="PyPI",
                    scope="dev",
                    manifest_path=str(path),
                    package_manager="pip",
                    version_spec_raw=spec,
                    exact_version=_detect_exact_version(spec),
                )
            )

        # Poetry: tool.poetry.dependencies = {name: "^1.2.3", ...}
        poetry = (payload.get("tool") or {}).get("poetry") or {}
        for name, val in (poetry.get("dependencies") or {}).items():
            n = _normalize_dep_name(str(name))
            if not n or n == "python":
                continue
            # v1: only string versions become version_spec_raw; tables become package-only
            spec = str(val).strip() if isinstance(val, str) else ""
            runtime_items.append(
                DependencyPackageRef(
                    name=n,
                    ecosystem="PyPI",
                    scope="runtime",
                    manifest_path=str(path),
                    package_manager="pip",
                    version_spec_raw=spec,
                    exact_version=_detect_exact_version(spec),
                )
            )

        poetry_dev = ((poetry.get("group") or {}).get("dev") or {}).get("dependencies") or {}
        for name, val in poetry_dev.items():
            n = _normalize_dep_name(str(name))
            if not n or n == "python":
                continue
            spec = str(val).strip() if isinstance(val, str) else ""
            dev_items.append(
                DependencyPackageRef(
                    name=n,
                    ecosystem="PyPI",
                    scope="dev",
                    manifest_path=str(path),
                    package_manager="pip",
                    version_spec_raw=spec,
                    exact_version=_detect_exact_version(spec),
                )
            )

        return runtime_items, dev_items

    @staticmethod
    def _inventory_requirements(path: Path) -> list[DependencyPackageRef]:
        items: list[DependencyPackageRef] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("-"):
                continue
            name, spec = _parse_python_req_line(s)
            if not name:
                continue
            items.append(
                DependencyPackageRef(
                    name=name,
                    ecosystem="PyPI",
                    scope="runtime",
                    manifest_path=str(path),
                    package_manager="pip",
                    version_spec_raw=spec,
                    exact_version=_detect_exact_version(spec),
                )
            )
        return items

    @staticmethod
    def _inventory_pipfile(path: Path) -> tuple[list[DependencyPackageRef], list[DependencyPackageRef]]:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        runtime_items: list[DependencyPackageRef] = []
        dev_items: list[DependencyPackageRef] = []

        for name, val in (payload.get("packages") or {}).items():
            n = _normalize_dep_name(str(name))
            spec = str(val).strip() if isinstance(val, str) else ""
            runtime_items.append(
                DependencyPackageRef(
                    name=n,
                    ecosystem="PyPI",
                    scope="runtime",
                    manifest_path=str(path),
                    package_manager="pip",
                    version_spec_raw=spec,
                    exact_version=_detect_exact_version(spec),
                )
            )

        for name, val in (payload.get("dev-packages") or {}).items():
            n = _normalize_dep_name(str(name))
            spec = str(val).strip() if isinstance(val, str) else ""
            dev_items.append(
                DependencyPackageRef(
                    name=n,
                    ecosystem="PyPI",
                    scope="dev",
                    manifest_path=str(path),
                    package_manager="pip",
                    version_spec_raw=spec,
                    exact_version=_detect_exact_version(spec),
                )
            )

        return runtime_items, dev_items

    @staticmethod
    def _inventory_package_json(path: Path) -> tuple[list[DependencyPackageRef], list[DependencyPackageRef]]:
        payload = _read_json(path)
        runtime_items: list[DependencyPackageRef] = []
        dev_items: list[DependencyPackageRef] = []

        for name, spec in (payload.get("dependencies") or {}).items():
            n = _normalize_dep_name(str(name))
            raw = str(spec).strip()
            runtime_items.append(
                DependencyPackageRef(
                    name=n,
                    ecosystem="npm",
                    scope="runtime",
                    manifest_path=str(path),
                    package_manager="npm",
                    version_spec_raw=raw,
                    exact_version=_detect_exact_version(raw),
                )
            )

        for name, spec in (payload.get("devDependencies") or {}).items():
            n = _normalize_dep_name(str(name))
            raw = str(spec).strip()
            dev_items.append(
                DependencyPackageRef(
                    name=n,
                    ecosystem="npm",
                    scope="dev",
                    manifest_path=str(path),
                    package_manager="npm",
                    version_spec_raw=raw,
                    exact_version=_detect_exact_version(raw),
                )
            )

        return runtime_items, dev_items

    @staticmethod
    def _inventory_cargo(path: Path) -> tuple[list[DependencyPackageRef], list[DependencyPackageRef]]:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        runtime_items: list[DependencyPackageRef] = []
        dev_items: list[DependencyPackageRef] = []

        for name, val in (payload.get("dependencies") or {}).items():
            n = _normalize_dep_name(str(name))
            spec = str(val).strip() if isinstance(val, str) else ""
            runtime_items.append(
                DependencyPackageRef(
                    name=n,
                    ecosystem="crates.io",
                    scope="runtime",
                    manifest_path=str(path),
                    package_manager="cargo",
                    version_spec_raw=spec,
                    exact_version=_detect_exact_version(spec),
                )
            )

        for name, val in (payload.get("dev-dependencies") or {}).items():
            n = _normalize_dep_name(str(name))
            spec = str(val).strip() if isinstance(val, str) else ""
            dev_items.append(
                DependencyPackageRef(
                    name=n,
                    ecosystem="crates.io",
                    scope="dev",
                    manifest_path=str(path),
                    package_manager="cargo",
                    version_spec_raw=spec,
                    exact_version=_detect_exact_version(spec),
                )
            )

        return runtime_items, dev_items

    @staticmethod
    def _inventory_go_mod(path: Path) -> list[DependencyPackageRef]:
        items: list[DependencyPackageRef] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(("module ", "go ", "(", ")", "//")):
                continue
            # lines in require blocks: "github.com/x/y v1.2.3"
            parts = stripped.split()
            if len(parts) >= 2 and "." in parts[0]:
                name = _normalize_dep_name(parts[0])
                ver = parts[1].strip()
                items.append(
                    DependencyPackageRef(
                        name=name,
                        ecosystem="Go",
                        scope="runtime",
                        manifest_path=str(path),
                        package_manager="go",
                        version_spec_raw=ver,
                        exact_version=_detect_exact_version(ver),
                    )
                )
        return items

    @staticmethod
    def _inventory_pubspec(path: Path) -> tuple[list[DependencyPackageRef], list[DependencyPackageRef]]:
        runtime: list[DependencyPackageRef] = []
        dev: list[DependencyPackageRef] = []
        section = ""
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped in {"dependencies:", "dev_dependencies:"}:
                section = stripped[:-1]
                continue
            if section and ":" in stripped and not stripped.startswith("-"):
                name, spec = stripped.split(":", 1)
                n = _normalize_dep_name(name)
                raw = spec.strip().strip("'").strip('"')
                if not n:
                    continue
                item = DependencyPackageRef(
                    name=n,
                    ecosystem="Pub",
                    scope=("runtime" if section == "dependencies" else "dev"),
                    manifest_path=str(path),
                    package_manager="pub",
                    version_spec_raw=raw,
                    exact_version=_detect_exact_version(raw),
                )
                (runtime if section == "dependencies" else dev).append(item)
        return runtime, dev

    @staticmethod
    def _inventory_composer(path: Path) -> tuple[list[DependencyPackageRef], list[DependencyPackageRef]]:
        payload = _read_json(path)
        runtime_items: list[DependencyPackageRef] = []
        dev_items: list[DependencyPackageRef] = []
        for name, spec in (payload.get("require") or {}).items():
            n = _normalize_dep_name(str(name))
            raw = str(spec).strip()
            runtime_items.append(
                DependencyPackageRef(
                    name=n,
                    ecosystem="Packagist",
                    scope="runtime",
                    manifest_path=str(path),
                    package_manager="composer",
                    version_spec_raw=raw,
                    exact_version=_detect_exact_version(raw),
                )
            )
        for name, spec in (payload.get("require-dev") or {}).items():
            n = _normalize_dep_name(str(name))
            raw = str(spec).strip()
            dev_items.append(
                DependencyPackageRef(
                    name=n,
                    ecosystem="Packagist",
                    scope="dev",
                    manifest_path=str(path),
                    package_manager="composer",
                    version_spec_raw=raw,
                    exact_version=_detect_exact_version(raw),
                )
            )
        return runtime_items, dev_items

    @staticmethod
    def _inventory_gemfile(path: Path) -> list[DependencyPackageRef]:
        # v1: name-only, version might exist but is diverse; leave package-only
        items: list[DependencyPackageRef] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped.startswith("gem "):
                continue
            match = re.search(r"gem\s+['\"]([^'\"]+)['\"]", stripped)
            if not match:
                continue
            n = _normalize_dep_name(match.group(1))
            if not n:
                continue
            items.append(
                DependencyPackageRef(
                    name=n,
                    ecosystem="RubyGems",
                    scope="runtime",
                    manifest_path=str(path),
                    package_manager="gem",
                    version_spec_raw="",
                    exact_version=None,
                )
            )
        return items
    
    def _manifest_deps(self, manifest: Path) -> tuple[list[str], list[str]]:
        """
        Legacy dependency name-only parser used by analyze().
        Keep this unchanged so DependencyService.analyze() keeps working.
        """
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
    
    def _manifest_inventory(self, manifest: Path) -> tuple[list[DependencyPackageRef], list[DependencyPackageRef]]:
        if manifest.name == "pyproject.toml":
            return self._inventory_pyproject(manifest)
        if manifest.name.startswith("requirements"):
            return self._inventory_requirements(manifest), []
        if manifest.name == "Pipfile":
            return self._inventory_pipfile(manifest)
        if manifest.name == "package.json":
            return self._inventory_package_json(manifest)
        if manifest.name == "pubspec.yaml":
            return self._inventory_pubspec(manifest)
        if manifest.name == "Cargo.toml":
            return self._inventory_cargo(manifest)
        if manifest.name == "go.mod":
            return self._inventory_go_mod(manifest), []
        if manifest.name == "composer.json":
            return self._inventory_composer(manifest)
        if manifest.name == "Gemfile":
            return self._inventory_gemfile(manifest), []
        # setup.py: too messy in v1, we keep name-only via analyze() graph; inventory skips it (warn upstream if desired)
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

    def collect_inventory(self, target_path: str | Path) -> list[DependencyPackageRef]:
        root = Path(target_path).resolve()
        if root.is_file():
            root = root.parent

        subprojects = discover_subprojects(root)

        items: list[DependencyPackageRef] = []
        for subproject in subprojects:
            for manifest in subproject.manifest_paths:
                runtime, dev = self._manifest_inventory(manifest)

                # make manifest_path relative to the root for report stability
                for ref in runtime:
                    ref.manifest_path = str(manifest.relative_to(root))
                for ref in dev:
                    ref.manifest_path = str(manifest.relative_to(root))

                items.extend(runtime)
                items.extend(dev)

        # Deduplicate per (ecosystem, name, scope, manifest_path, version_spec_raw)
        seen: set[tuple[str, str, str, str, str]] = set()
        deduped: list[DependencyPackageRef] = []
        for it in items:
            key = (it.ecosystem, it.name, it.scope, it.manifest_path, it.version_spec_raw)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(it)

        return sorted(deduped, key=lambda d: (d.ecosystem, d.name, d.scope, d.manifest_path))