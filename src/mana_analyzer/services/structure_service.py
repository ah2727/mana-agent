from __future__ import annotations

from pathlib import Path
from typing import Any
from mana_analyzer.analysis.models import (
    ClassDescriptor,
    ExportDescriptor,
    ModuleDescriptor,
    ProjectStructureReport,
    SubprojectReport,
)
from mana_analyzer.services.dependency_service import DependencyService
from mana_analyzer.services.parsers import (
    ParsedModule,
    parse_dart_module,
    parse_js_ts_module,
    parse_jvm_module,
    parse_markup_module,
    parse_native_module,
    parse_python_module,
    parse_scripting_module,
)
from mana_analyzer.utils.io import EXCLUDED_DIRS, iter_source_files, language_for_path, load_ignore_patterns
from mana_analyzer.utils.project_discovery import discover_subprojects


class StructureService:
    def __init__(self, include_tests: bool = False) -> None:
        self.include_tests = include_tests

    @staticmethod
    def _parse_module(file_path: Path, project_root: Path, language: str) -> ParsedModule:
        suffix = file_path.suffix.lower()
        if language == "python":
            return parse_python_module(file_path, project_root)
        if suffix in {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}:
            return parse_js_ts_module(file_path, project_root)
        if suffix == ".dart":
            return parse_dart_module(file_path, project_root)
        if suffix in {".java", ".kt"}:
            return parse_jvm_module(file_path, project_root)
        if suffix in {".swift", ".m", ".mm", ".c", ".cc", ".cpp", ".h", ".hpp", ".cs", ".scala", ".rs", ".go"}:
            return parse_native_module(file_path, project_root)
        if suffix in {".sh", ".bash", ".zsh", ".php", ".rb", ".sql"}:
            return parse_scripting_module(file_path, project_root)
        return parse_markup_module(file_path, project_root)

    @staticmethod
    def _list_directories(project_root: Path) -> list[str]:
        ignore_patterns = load_ignore_patterns(project_root)
        directories: list[str] = []
        for path in project_root.rglob("*"):
            if not path.is_dir():
                continue
            relative = str(path.relative_to(project_root))
            if not relative:
                continue
            if any(part in EXCLUDED_DIRS for part in path.parts):
                continue
            if any(relative == pattern.rstrip("/") or relative.startswith(pattern.rstrip("/") + "/") for pattern in ignore_patterns):
                continue
            directories.append(relative)
        return sorted(set(directories))

    def analyze_project(self, target_path: str | Path) -> ProjectStructureReport:
        target = Path(target_path).resolve()
        project_root = target if target.is_dir() else target.parent

        dependency_report = DependencyService().analyze(project_root)

        files = iter_source_files(project_root)
        if not self.include_tests:
            files = [item for item in files if "tests" not in item.parts and "__tests__" not in item.parts]
        all_file_paths = []
        for file_path in files:
            module_path = str(file_path.relative_to(project_root)).replace("\\", "/")
            all_file_paths.append(module_path)
        all_file_paths = sorted(set(all_file_paths))
        modules: list[ModuleDescriptor] = []
        exports: list[ExportDescriptor] = []
        data_structures: list[ClassDescriptor] = []
        commands: list[str] = []
        import_roots: set[str] = set()

        files_by_language: dict[str, list[str]] = {}
        for file_path in files:
            module_path = str(file_path.relative_to(project_root))
            language = language_for_path(file_path)
            parsed = self._parse_module(file_path, project_root, language)

            modules.append(
                ModuleDescriptor(
                    module_path=module_path,
                    imports=parsed.imports,
                    functions=parsed.functions,
                    classes=parsed.classes,
                    constants=parsed.constants,
                    language=language,
                    parse_mode=parsed.parse_mode,
                )
            )
            exports.extend(parsed.exports)
            data_structures.extend(parsed.data_structures)
            commands.extend(parsed.commands)
            import_roots.update(parsed.import_roots)
            files_by_language.setdefault(language, []).append(module_path)

        language_counts = {key: len(value) for key, value in sorted(files_by_language.items())}
        files_by_language = {key: sorted(value) for key, value in sorted(files_by_language.items())}

        ci_files: list[str] = []
        workflows = project_root / ".github" / "workflows"
        if workflows.exists():
            ci_files = sorted(str(item.relative_to(project_root)) for item in workflows.glob("*.y*ml"))

        llm_capabilities = [
            "qna-chain",
            "llm-static-analysis",
            "semantic-search-rag",
            "agent-tools-opt-in",
        ]

        subprojects = discover_subprojects(project_root)
        subproject_reports = [
            SubprojectReport(
                root_path=str(item.root_path.relative_to(project_root)),
                manifest_paths=sorted(str(path.relative_to(project_root)) for path in item.manifest_paths),
                package_managers=item.package_managers,
                framework_hints=item.framework_hints,
            )
            for item in subprojects
        ]

        package_manager = ", ".join(dependency_report.package_managers) if dependency_report.package_managers else "pip/setuptools"

        return ProjectStructureReport(
            project_root=str(project_root),
            frameworks=dependency_report.frameworks,
            runtime="Python 3.12",
            package_manager=package_manager,
            entrypoints=[],
            ci=ci_files,
            tech_stack=dependency_report.technologies,
            dependencies_runtime=dependency_report.runtime_dependencies,
            dependencies_dev=dependency_report.dev_dependencies,
            modules=sorted(modules, key=lambda item: item.module_path),
            exports=sorted(exports, key=lambda item: (item.source_module, item.symbol, item.mechanism)),
            data_structures=sorted(data_structures, key=lambda item: item.name),
            commands=sorted(set(commands)),
            llm_capabilities=llm_capabilities,
            subprojects=subproject_reports,
            directories=self._list_directories(project_root),
            files_by_language=files_by_language,
            language_counts=language_counts,
            files=all_file_paths,
            file_counts={"total_files": len(all_file_paths)},
            discovery_stats={
                "scope": "source+config",
                "excluded_dir_names": sorted(EXCLUDED_DIRS),
                "ignored_patterns_applied": [
                    name for name in [".gitignore", ".aiignore"] if (project_root / name).exists()
                ],
                "include_tests": self.include_tests,
            },
        )

    @staticmethod
    def render_markdown(report: ProjectStructureReport) -> str:
        lines: list[str] = []
        lines.append("# Project Structure Analysis")
        lines.append("")
        lines.append("## Architecture")
        lines.append(f"- Project root: `{report.project_root}`")
        lines.append(f"- Runtime: `{report.runtime}`")
        lines.append(f"- Package manager: `{report.package_manager}`")
        lines.append(f"- Frameworks: {', '.join(report.frameworks) if report.frameworks else 'none'}")
        lines.append("")
        lines.append("## Stack")
        lines.append(f"- Tech: {', '.join(report.tech_stack) if report.tech_stack else 'none'}")
        lines.append(f"- Runtime dependencies: {len(report.dependencies_runtime)}")
        lines.append(f"- Dev dependencies: {len(report.dependencies_dev)}")
        lines.append("")
        lines.append("## Languages")
        if report.language_counts:
            for language, count in report.language_counts.items():
                lines.append(f"- `{language}`: {count}")
        else:
            lines.append("- none")
        lines.append("")
        lines.append("## Directory Tree")
        if report.directories:
            for directory in report.directories:
                lines.append(f"- `{directory}`")
        else:
            lines.append("- none")
        lines.append("")
        lines.append("## Modules")
        for module in report.modules:
            lines.append(
                f"- `{module.module_path}` lang={module.language} parse={module.parse_mode} funcs={len(module.functions)} classes={len(module.classes)} imports={len(module.imports)}"
            )
        lines.append("")
        lines.append("## APIs and Exports")
        for export in report.exports:
            lines.append(f"- `{export.source_module}` `{export.symbol}` via `{export.mechanism}`")
        lines.append("")
        lines.append("## Data Structures")
        for class_desc in report.data_structures:
            lines.append(
                f"- `{class_desc.name}` fields={len(class_desc.fields)} methods={len(class_desc.methods)} decorators={','.join(class_desc.decorators) or 'none'}"
            )
        lines.append("")
        lines.append("## Command Surface")
        for command in report.commands:
            lines.append(f"- `{command}`")
        lines.append("")
        lines.append("## LLM and Tooling")
        for capability in report.llm_capabilities:
            lines.append(f"- `{capability}`")
        return "\n".join(lines)

    def render_file_tree_markdown(self, files: list[str]) -> str:
        tree: dict[str, Any] = {}
        for p in files:
            parts = [x for x in p.replace("\\", "/").split("/") if x]
            cur = tree
            for part in parts[:-1]:
                cur = cur.setdefault(part + "/", {})
            cur.setdefault(parts[-1], None)

        lines: list[str] = ["```text"]

        def walk(node: dict[str, Any], prefix: str = "") -> None:
            keys = sorted(node.keys(), key=lambda k: (0 if k.endswith("/") else 1, k))
            for i, k in enumerate(keys):
                last = i == len(keys) - 1
                branch = "└── " if last else "├── "
                lines.append(prefix + branch + k.rstrip("/"))
                child = node[k]
                if isinstance(child, dict):
                    ext = "    " if last else "│   "
                    walk(child, prefix + ext)

        walk(tree)
        lines.append("```")
        return "\n".join(lines)

    def compute_hotspots(self, report: ProjectStructureReport, top_n: int = 15) -> list[dict[str, Any]]:
        # Export counts per module
        exports_by_module: dict[str, int] = {}
        for e in report.exports:
            exports_by_module[e.source_module] = exports_by_module.get(e.source_module, 0) + 1

        # Command presence is global list; we can treat modules with many commands as higher signal only if we can map them.
        # Since commands are strings, we do not attribute them to files (v1 deterministic simplification).

        scored: list[tuple[str, int, str]] = []
        for m in report.modules:
            imports_n = len(m.imports)
            funcs_n = len(m.functions)
            classes_n = len(m.classes)
            exports_n = exports_by_module.get(m.module_path, 0)

            score = imports_n + funcs_n + (2 * classes_n) + (2 * exports_n)
            reason_parts = []
            if imports_n:
                reason_parts.append(f"imports={imports_n}")
            if exports_n:
                reason_parts.append(f"exports={exports_n}")
            if funcs_n:
                reason_parts.append(f"functions={funcs_n}")
            if classes_n:
                reason_parts.append(f"classes={classes_n}")

            reason = " / ".join(reason_parts) if reason_parts else "structure present"
            scored.append((m.module_path, score, reason))

        scored.sort(key=lambda t: (-t[1], t[0]))
        top = scored[:top_n] if scored else [(p, 1, "file present") for p in (report.files[:top_n] if report.files else [])]

        return [{"path": path, "score": int(score), "reason": reason} for path, score, reason in top]