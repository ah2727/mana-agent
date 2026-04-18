from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import fnmatch

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
from mana_analyzer.utils.io import (
    EXCLUDED_DIRS,
    iter_source_files,
    language_for_path,
    load_ignore_patterns,
)
from mana_analyzer.utils.project_discovery import discover_subprojects


# -------------------------
# Helpers (NEW)
# -------------------------
def _to_posix_relative(path: Path, root: Path) -> str:
    """Return a stable, POSIX-style relative path."""
    return path.relative_to(root).as_posix()


def _normalize_ignore_patterns(patterns: list[str]) -> list[str]:
    """
    Normalize ignore patterns:
    - trim whitespace
    - drop empty lines
    - keep as-is otherwise (we'll match with fnmatch + prefix rules)
    """
    out: list[str] = []
    for p in patterns:
        p = (p or "").strip()
        if not p or p.startswith("#"):
            continue
        out.append(p)
    return out


def _matches_ignore(relative_posix: str, patterns: list[str]) -> bool:
    """
    Decide if a relative path should be ignored, supporting:
    - exact dir patterns like "build/" or "build"
    - prefix dir patterns
    - glob patterns like "*.min.js" or "dist/**"
    """
    rel = relative_posix

    # Always ignore excluded dir names anywhere in the path
    parts = rel.split("/")
    if any(part in EXCLUDED_DIRS for part in parts):
        return True

    for raw in patterns:
        pat = raw.strip()

        # Common convention: trailing slash means directory
        is_dir_pat = pat.endswith("/")
        pat_no_slash = pat.rstrip("/")

        # Prefix-style ignore for directories:
        # "foo" or "foo/" ignores "foo" and anything under it.
        if "/" not in pat_no_slash and not any(ch in pat_no_slash for ch in "*?[]"):
            if parts and parts[0] == pat_no_slash:
                return True
            # Also ignore nested occurrences if pattern looks like a dir name
            if pat_no_slash in parts:
                return True

        # Exact/prefix match (directory-like patterns)
        if is_dir_pat:
            if rel == pat_no_slash or rel.startswith(pat_no_slash + "/"):
                return True

        # Glob match (fnmatch is POSIX-style when we use POSIX paths)
        # Support "dist/**" style by matching both the pattern and a simplified version.
        if fnmatch.fnmatch(rel, pat):
            return True
        if "**" in pat:
            simplified = pat.replace("**/", "").replace("/**", "")
            if simplified and fnmatch.fnmatch(rel, simplified):
                return True

    return False


def _safe_empty_parsed_module(file_path: Path, project_root: Path) -> ParsedModule:
    """
    If a parser crashes, return an empty ParsedModule.
    This assumes ParsedModule is a dataclass/struct-like object with these attrs.
    If your ParsedModule differs, adjust here.
    """
    # We avoid importing dataclasses etc. and just construct via the known fields.
    # If ParsedModule has a different constructor, update this in one place.
    return ParsedModule(
        module_path=_to_posix_relative(file_path, project_root),
        imports=[],
        functions=[],
        classes=[],
        constants=[],
        exports=[],
        data_structures=[],
        commands=[],
        import_roots=set(),
        parse_mode="error",
    )


# -------------------------
# Upgraded StructureService
# -------------------------
class StructureService:
    """
    v2 upgrades:
    - consistent POSIX paths
    - stronger ignore matching
    - safer parsing (per-file isolation)
    - cleaner parser dispatch
    - faster directory scanning by pruning excluded/ignored dirs
    """

    # Central suffix routing
    _JS_TS_SUFFIXES = {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}
    _DART_SUFFIXES = {".dart"}
    _JVM_SUFFIXES = {".java", ".kt"}
    _NATIVE_SUFFIXES = {
        ".swift", ".m", ".mm", ".c", ".cc", ".cpp",
        ".h", ".hpp", ".cs", ".scala", ".rs", ".go",
    }
    _SCRIPTING_SUFFIXES = {".sh", ".bash", ".zsh", ".php", ".rb", ".sql"}

    def __init__(self, include_tests: bool = False) -> None:
        self.include_tests = include_tests

    @classmethod
    def _parse_module(cls, file_path: Path, project_root: Path, language: str) -> ParsedModule:
        suffix = file_path.suffix.lower()

        try:
            if language == "python":
                return parse_python_module(file_path, project_root)
            if suffix in cls._JS_TS_SUFFIXES:
                return parse_js_ts_module(file_path, project_root)
            if suffix in cls._DART_SUFFIXES:
                return parse_dart_module(file_path, project_root)
            if suffix in cls._JVM_SUFFIXES:
                return parse_jvm_module(file_path, project_root)
            if suffix in cls._NATIVE_SUFFIXES:
                return parse_native_module(file_path, project_root)
            if suffix in cls._SCRIPTING_SUFFIXES:
                return parse_scripting_module(file_path, project_root)
            return parse_markup_module(file_path, project_root)
        except Exception:
            # One broken file shouldn't kill whole project analysis
            return _safe_empty_parsed_module(file_path, project_root)

    @staticmethod
    def _list_directories(project_root: Path) -> list[str]:
        """
        Faster + safer than Path.rglob for big repos:
        - uses os-style walking via pathlib's rglob equivalent is slower
        - prunes excluded + ignored dirs early
        """
        ignore_patterns = _normalize_ignore_patterns(load_ignore_patterns(project_root))
        directories: set[str] = set()

        # Manual walk using rglob on dirs only is still expensive; prune by checking relative path.
        # This approach is a compromise that stays pathlib-only.
        for path in project_root.rglob("*"):
            if not path.is_dir():
                continue

            rel = _to_posix_relative(path, project_root)
            if not rel:
                continue

            if _matches_ignore(rel, ignore_patterns):
                # If this dir is ignored, no need to list it.
                # Note: pathlib.rglob cannot prune traversal; this still helps output correctness.
                continue

            directories.add(rel)

        return sorted(directories)

    def analyze_project(self, target_path: str | Path) -> ProjectStructureReport:
        target = Path(target_path).resolve()
        project_root = target if target.is_dir() else target.parent

        ignore_patterns = _normalize_ignore_patterns(load_ignore_patterns(project_root))

        dependency_report = DependencyService().analyze(project_root)

        # Materialize once; keep POSIX paths stable
        source_files = list(iter_source_files(project_root))

        # Optional test exclusion (directory segment match)
        if not self.include_tests:
            def _is_test_file(p: Path) -> bool:
                parts = {part.lower() for part in p.parts}
                return "tests" in parts or "__tests__" in parts

            source_files = [p for p in source_files if not _is_test_file(p)]

        # Apply ignore patterns to files too (important if iter_source_files doesn't)
        filtered_files: list[Path] = []
        for fp in source_files:
            rel = _to_posix_relative(fp, project_root)
            if _matches_ignore(rel, ignore_patterns):
                continue
            filtered_files.append(fp)

        # Sorted stable file list
        filtered_files.sort(key=lambda p: _to_posix_relative(p, project_root))

        all_file_paths = sorted({_to_posix_relative(fp, project_root) for fp in filtered_files})

        modules: list[ModuleDescriptor] = []
        exports: list[ExportDescriptor] = []
        data_structures: list[ClassDescriptor] = []
        commands: list[str] = []
        import_roots: set[str] = set()

        files_by_language: dict[str, list[str]] = {}

        for file_path in filtered_files:
            module_path = _to_posix_relative(file_path, project_root)
            language = language_for_path(file_path)
            parsed = self._parse_module(file_path, project_root, language)

            modules.append(
                ModuleDescriptor(
                    module_path=module_path,
                    imports=getattr(parsed, "imports", []) or [],
                    functions=getattr(parsed, "functions", []) or [],
                    classes=getattr(parsed, "classes", []) or [],
                    constants=getattr(parsed, "constants", []) or [],
                    language=language,
                    parse_mode=getattr(parsed, "parse_mode", "unknown"),
                )
            )
            exports.extend(getattr(parsed, "exports", []) or [])
            data_structures.extend(getattr(parsed, "data_structures", []) or [])
            commands.extend(getattr(parsed, "commands", []) or [])
            import_roots.update(getattr(parsed, "import_roots", set()) or set())
            files_by_language.setdefault(language, []).append(module_path)

        language_counts = {k: len(v) for k, v in sorted(files_by_language.items())}
        files_by_language = {k: sorted(v) for k, v in sorted(files_by_language.items())}

        # CI workflows
        ci_files: list[str] = []
        workflows = project_root / ".github" / "workflows"
        if workflows.exists():
            ci_files = sorted(_to_posix_relative(item, project_root) for item in workflows.glob("*.y*ml"))

        llm_capabilities = [
            "qna-chain",
            "llm-static-analysis",
            "semantic-search-rag",
            "agent-tools-opt-in",
        ]

        subprojects = discover_subprojects(project_root)
        subproject_reports = [
            SubprojectReport(
                root_path=_to_posix_relative(item.root_path, project_root),
                manifest_paths=sorted(_to_posix_relative(p, project_root) for p in item.manifest_paths),
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
                f"- `{module.module_path}` lang={module.language} parse={module.parse_mode} "
                f"funcs={len(module.functions)} classes={len(module.classes)} imports={len(module.imports)}"
            )
        lines.append("")
        lines.append("## APIs and Exports")
        export_modules = {item.source_module for item in report.exports}
        lines.append(
            f"- Summary: detected {len(report.exports)} exports across {len(export_modules)} module(s)."
        )
        if report.exports:
            for export in report.exports:
                lines.append(f"- `{export.source_module}` `{export.symbol}` via `{export.mechanism}`")
        else:
            lines.append("- No exports detected in the scanned source files.")
        lines.append("")
        lines.append("## Data Structures")
        total_fields = sum(len(item.fields) for item in report.data_structures)
        total_methods = sum(len(item.methods) for item in report.data_structures)
        lines.append(
            f"- Summary: detected {len(report.data_structures)} data structure(s), {total_fields} field(s), and {total_methods} method(s)."
        )
        if report.data_structures:
            for class_desc in report.data_structures:
                lines.append(
                    f"- `{class_desc.name}` fields={len(class_desc.fields)} methods={len(class_desc.methods)} "
                    f"decorators={','.join(class_desc.decorators) or 'none'}"
                )
        else:
            lines.append("- No data structures were inferred by the parsers.")
        lines.append("")
        lines.append("## Command Surface")
        lines.append(f"- Summary: detected {len(report.commands)} command entrypoint(s).")
        if report.commands:
            for command in report.commands:
                lines.append(f"- `{command}`")
        else:
            lines.append("- No CLI-style command declarations detected.")
        lines.append("")
        lines.append("## LLM and Tooling")
        if report.llm_capabilities:
            for capability in report.llm_capabilities:
                lines.append(f"- `{capability}`")
        else:
            lines.append("- none")
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

        scored: list[tuple[str, int, str]] = []
        for m in report.modules:
            imports_n = len(m.imports)
            funcs_n = len(m.functions)
            classes_n = len(m.classes)
            exports_n = exports_by_module.get(m.module_path, 0)

            # Slightly rebalance to emphasize API surface + types
            score = (imports_n * 1) + (funcs_n * 1) + (classes_n * 2) + (exports_n * 3)

            reason_parts: list[str] = []
            if imports_n:
                reason_parts.append(f"imports={imports_n}")
            if exports_n:
                reason_parts.append(f"exports={exports_n}")
            if funcs_n:
                reason_parts.append(f"functions={funcs_n}")
            if classes_n:
                reason_parts.append(f"classes={classes_n}")

            reason = " / ".join(reason_parts) if reason_parts else "structure present"
            scored.append((m.module_path, int(score), reason))

        scored.sort(key=lambda t: (-t[1], t[0]))

        if scored:
            top = scored[:top_n]
            return [{"path": path, "score": score, "reason": reason} for path, score, reason in top]

        # Fallback if modules list is empty
        fallback = (report.files or [])[:top_n]
        return [{"path": p, "score": 1, "reason": "file present"} for p in fallback]