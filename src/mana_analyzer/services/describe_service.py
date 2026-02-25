from __future__ import annotations

import ast
import fnmatch
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from mana_analyzer.analysis.models import CodeDescription, DependencyGraphReport, DescribeReport
from mana_analyzer.llm.repo_chain import RepositoryMultiChain
from mana_analyzer.services.dependency_service import DependencyService
from mana_analyzer.utils.io import iter_source_files, language_for_path

logger = logging.getLogger(__name__)


class DescribeService:
    _DEFAULT_MAX_SOURCE_CHARS = 12000
    _CHUNK_SIZE = 6000
    _CHUNK_OVERLAP = 600

    def __init__(self, dependency_service: DependencyService, llm_chain: RepositoryMultiChain | None = None) -> None:
        self.dependency_service = dependency_service
        self.llm_chain = llm_chain

    @staticmethod
    def _language_from_suffix(path: Path) -> str:
        language = language_for_path(path)
        return "text" if language == "unknown" else language

    @staticmethod
    def _safe_mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    @staticmethod
    def _safe_size(path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0

    @classmethod
    def _module_name(cls, root: Path, path: Path) -> str:
        relative = path.relative_to(root)
        parts = list(relative.parts)
        if parts and parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
        elif parts:
            parts[-1] = Path(parts[-1]).stem
        if parts and parts[-1] in {"__init__", "index"}:
            parts = parts[:-1]
        if not parts:
            return str(relative)
        return ".".join(parts)

    @staticmethod
    def _is_entrypoint(path: Path, source: str) -> bool:
        if path.suffix == ".py" and "if __name__ == \"__main__\"" in source:
            return True
        if path.name == "package.json":
            try:
                payload = json.loads(source)
            except json.JSONDecodeError:
                return False
            scripts = payload.get("scripts")
            return isinstance(scripts, dict) and bool(scripts)
        if path.name in {"main.py", "app.py", "main.ts", "main.js", "server.js"}:
            return True
        return False

    @staticmethod
    def _extract_python_symbols(source: str, include_functions: bool) -> tuple[list[str], dict[str, str]]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return [], {}
        symbols: list[str] = []
        docs: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                symbols.append(node.name)
                docs[node.name] = (ast.get_docstring(node) or "").strip()
                if include_functions:
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            name = f"{node.name}.{item.name}"
                            symbols.append(name)
                            docs[name] = (ast.get_docstring(item) or "").strip()
            elif include_functions and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(node.name)
                docs[node.name] = (ast.get_docstring(node) or "").strip()
        return sorted(set(symbols)), {k: v for k, v in docs.items() if v}

    @staticmethod
    def _extract_regex_symbols(
        source: str,
        patterns: list[tuple[str, str]],
    ) -> list[str]:
        symbols: set[str] = set()
        for pattern, prefix in patterns:
            for match in re.finditer(pattern, source, flags=re.MULTILINE):
                name = match.group(1)
                if name:
                    symbols.add(f"{prefix}{name}" if prefix else name)
        return sorted(symbols)

    @classmethod
    def _heuristic_symbols(
        cls,
        path: Path,
        source: str,
        include_functions: bool,
    ) -> tuple[list[str], dict[str, str]]:
        suffix = path.suffix.lower()
        if suffix == ".py":
            return cls._extract_python_symbols(source, include_functions=include_functions)

        if suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
            patterns = [(r"(?:class|interface|type)\s+([A-Za-z_][A-Za-z0-9_]*)", "")]
            if include_functions:
                patterns.extend(
                    [
                        (r"function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", ""),
                        (r"const\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\(", ""),
                        (r"export\s+default\s+function\s+([A-Za-z_][A-Za-z0-9_]*)", ""),
                    ]
                )
            return cls._extract_regex_symbols(source, patterns), {}

        if suffix == ".go":
            patterns = [(r"type\s+([A-Za-z_][A-Za-z0-9_]*)\s+struct", "")]
            if include_functions:
                patterns.append((r"func\s+(?:\([^)]+\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\(", ""))
            return cls._extract_regex_symbols(source, patterns), {}

        if suffix in {".java", ".kt"}:
            patterns = [(r"(?:class|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)", "")]
            if include_functions:
                patterns.append((r"(?:public|private|protected|internal)?\s*(?:suspend\s+)?(?:fun|[A-Za-z0-9_<>,\[\] ?]+)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", ""))
            return cls._extract_regex_symbols(source, patterns), {}

        return [], {}

    @staticmethod
    def _extract_anchor_lines(path: Path, source: str, include_docstrings: bool) -> str:
        lines = source.splitlines()
        anchors: list[str] = []

        if path.suffix == ".py":
            for line in lines:
                stripped = line.strip()
                if stripped.startswith(("def ", "class ", "import ", "from ")):
                    anchors.append(stripped)
            if include_docstrings:
                in_triple = False
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        anchors.append(stripped)
                        in_triple = not in_triple if stripped.count('"""') % 2 == 1 or stripped.count("'''") % 2 == 1 else in_triple
                    elif in_triple and stripped:
                        anchors.append(stripped)
                        if len(anchors) >= 80:
                            break
        else:
            for line in lines:
                stripped = line.strip()
                if stripped.startswith(("function ", "class ", "interface ", "type ", "import ", "export ")):
                    anchors.append(stripped)
                elif include_docstrings and stripped.startswith(("//", "/*", "*")):
                    anchors.append(stripped)

        return "\n".join(anchors[:120])

    @classmethod
    def _split_text_for_llm(cls, text: str, max_chars: int) -> list[str]:
        if len(text) <= max_chars:
            return [text]

        chunks: list[str] = []
        start = 0
        step = max(1, cls._CHUNK_SIZE - cls._CHUNK_OVERLAP)
        while start < len(text):
            end = min(len(text), start + cls._CHUNK_SIZE)
            chunks.append(text[start:end])
            start += step
        return chunks

    def _summarize_with_llm(
        self,
        *,
        file_path: Path,
        language: str,
        source: str,
        anchors: str,
    ) -> tuple[str, list[str], bool]:
        if self.llm_chain is None:
            return "", [], False

        chunks = self._split_text_for_llm(source, self._DEFAULT_MAX_SOURCE_CHARS)
        chunk_summaries: list[str] = []
        merged_symbols: set[str] = set()

        for index, chunk in enumerate(chunks):
            prompt_source = chunk
            if anchors:
                prompt_source = f"# Anchors\n{anchors}\n\n# Code\n{chunk}"
            summary, llm_symbols = self.llm_chain.summarize_file(
                file_path=file_path,
                language=language,
                source=prompt_source[: self._DEFAULT_MAX_SOURCE_CHARS],
            )
            merged_symbols.update(llm_symbols)
            if len(chunks) == 1:
                return summary, sorted(merged_symbols), True
            chunk_summaries.append(f"Chunk {index + 1}: {summary}")

        combined = " ".join(chunk_summaries)
        return combined[:1400], sorted(merged_symbols), True

    @classmethod
    def _cache_path(cls, root: Path) -> Path:
        return root / ".mana_cache" / "describe_cache.json"

    @classmethod
    def _load_cache(cls, root: Path) -> dict[str, Any]:
        path = cls._cache_path(root)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    @classmethod
    def _save_cache(cls, root: Path, payload: dict[str, Any]) -> None:
        path = cls._cache_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def _matches_patterns(cls, relative: str, include: list[str] | None, exclude: list[str] | None) -> bool:
        normalized = relative.replace("\\", "/")
        if include and not any(fnmatch.fnmatch(normalized, pattern) for pattern in include):
            return False
        if exclude and any(fnmatch.fnmatch(normalized, pattern) for pattern in exclude):
            return False
        return True

    @classmethod
    def _dependency_centrality(cls, root: Path, dependency_report: DependencyGraphReport) -> dict[str, int]:
        counts: dict[str, int] = {}
        for edge in dependency_report.module_edges:
            counts[edge.source] = counts.get(edge.source, 0) + 1
            counts[edge.target] = counts.get(edge.target, 0) + 1

        by_file: dict[str, int] = {}
        for module, score in counts.items():
            module_path = Path(*module.split("."))
            for suffix in (".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".java", ".kt", ".rs", ".php", ".rb", ".c", ".cc", ".cpp"):
                candidate = root / (str(module_path) + suffix)
                if candidate.exists():
                    by_file[str(candidate.relative_to(root))] = score
                    break
        return by_file

    def _select_files(
        self,
        root: Path,
        dependency_report: DependencyGraphReport,
        max_files: int,
        include_patterns: list[str] | None,
        exclude_patterns: list[str] | None,
        modified_since: datetime | None,
    ) -> list[Path]:
        files = iter_source_files(root)
        centrality = self._dependency_centrality(root, dependency_report)
        scored: list[tuple[float, Path]] = []

        modified_since_ts = modified_since.timestamp() if modified_since else None

        for file_path in files:
            relative = str(file_path.relative_to(root))
            if not self._matches_patterns(relative, include_patterns, exclude_patterns):
                continue
            if modified_since_ts is not None and self._safe_mtime(file_path) < modified_since_ts:
                continue

            name = file_path.name.lower()
            score = 0.0
            if name in {"readme.md", "pyproject.toml", "package.json", "go.mod", "cargo.toml", "composer.json", "gemfile"}:
                score += 10
            if "config" in name or "settings" in name:
                score += 5
            if file_path.suffix in {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".java", ".kt"}:
                score += 4
            if name in {"main.py", "app.py", "main.ts", "main.js", "server.js"}:
                score += 8

            size_kb = self._safe_size(file_path) // 1024
            score += min(size_kb, 12)
            score += centrality.get(relative, 0) * 2
            scored.append((score, file_path))

        scored.sort(key=lambda item: (-item[0], str(item[1])))
        return [item for _, item in scored[: max(1, max_files)]]

    @staticmethod
    def _local_summary(path: Path, source: str, language: str, symbols: list[str], is_entrypoint: bool) -> str:
        lines = source.splitlines()
        entry = " entrypoint" if is_entrypoint else ""
        return (
            f"{path.name} is a {language}{entry} file with {len(lines)} lines and "
            f"{len(symbols)} discovered symbols."
        )

    @staticmethod
    def _read_sources(paths: list[Path]) -> dict[Path, str]:
        def _read(path: Path) -> tuple[Path, str]:
            return path, path.read_text(encoding="utf-8", errors="ignore")

        with ThreadPoolExecutor(max_workers=min(8, max(1, len(paths)))) as pool:
            return dict(pool.map(_read, paths))

    @staticmethod
    def _build_mermaid(dependency_report: DependencyGraphReport, max_edges: int = 60) -> str:
        lines = ["flowchart LR"]
        for edge in dependency_report.module_edges[:max_edges]:
            src = edge.source.replace('"', "")
            dst = edge.target.replace('"', "")
            lines.append(f'  "{src}" --> "{dst}"')
        if len(lines) == 1:
            lines.append('  "repository" --> "no-module-edges"')
        return "\n".join(lines)

    @staticmethod
    def _build_architecture_data(
        dependency_report: DependencyGraphReport,
        descriptions: list[CodeDescription],
    ) -> dict[str, Any]:
        modules: list[dict[str, Any]] = []
        for item in descriptions:
            modules.append(
                {
                    "file_path": item.file_path,
                    "language": item.language,
                    "symbols": item.symbols,
                    "entrypoint": item.entrypoint,
                }
            )
        return {
            "project_root": dependency_report.project_root,
            "package_managers": dependency_report.package_managers,
            "frameworks": dependency_report.frameworks,
            "technologies": dependency_report.technologies,
            "languages": dependency_report.languages,
            "runtime_dependencies": dependency_report.runtime_dependencies,
            "dev_dependencies": dependency_report.dev_dependencies,
            "module_edge_count": len(dependency_report.module_edges),
            "external_edge_count": len(dependency_report.dependency_edges),
            "modules": modules,
            "business_critical_modules": [
                item.file_path
                for item in sorted(
                    descriptions,
                    key=lambda row: (len(row.symbols), len(row.summary)),
                    reverse=True,
                )[:5]
            ],
        }

    def describe_async(self, *args: Any, **kwargs: Any) -> Any:
        import asyncio

        return asyncio.to_thread(self.describe, *args, **kwargs)

    def describe(
        self,
        target_path: str | Path,
        max_files: int = 12,
        include_functions: bool = False,
        use_llm: bool = True,
        model_override: str | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        modified_since: datetime | None = None,
        include_docstrings: bool = True,
        use_cache: bool = True,
        return_structured: bool = True,
    ) -> DescribeReport:
        _ = model_override
        started = perf_counter()
        root = Path(target_path).resolve()
        if root.is_file():
            root = root.parent

        if modified_since and modified_since.tzinfo is None:
            modified_since = modified_since.replace(tzinfo=timezone.utc)

        chain_steps = [
            "file-tree-selection",
            "dependency-analysis",
            "file-summarization",
            "architecture-synthesis",
        ]

        dependency_report = self.dependency_service.analyze(root)
        selected = self._select_files(
            root,
            dependency_report=dependency_report,
            max_files=max_files,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            modified_since=modified_since,
        )

        cache = self._load_cache(root) if use_cache else {}
        cached_files = cache.get("files", {}) if isinstance(cache.get("files", {}), dict) else {}

        source_by_file = self._read_sources(selected)
        descriptions: list[CodeDescription] = []
        cache_updates: dict[str, Any] = {}

        llm_attempts = 0
        llm_failures = 0
        cache_hits = 0

        for file_path in selected:
            relative = str(file_path.relative_to(root))
            source = source_by_file.get(file_path, "")
            language = self._language_from_suffix(file_path)
            symbols, symbol_docs = self._heuristic_symbols(file_path, source, include_functions=include_functions)
            entrypoint = self._is_entrypoint(file_path, source)

            mtime = self._safe_mtime(file_path)
            cached = cached_files.get(relative)
            if use_cache and isinstance(cached, dict) and cached.get("mtime") == mtime:
                descriptions.append(
                    CodeDescription(
                        file_path=relative,
                        language=language,
                        symbols=[str(item) for item in cached.get("symbols", [])],
                        summary=str(cached.get("summary", "")),
                        entrypoint=bool(cached.get("entrypoint", False)),
                        symbol_docs={str(k): str(v) for k, v in (cached.get("symbol_docs", {}) or {}).items()},
                    )
                )
                cache_updates[relative] = cached
                cache_hits += 1
                continue

            merged_symbols = symbols
            summary = self._local_summary(file_path, source, language, symbols, entrypoint)

            if use_llm and self.llm_chain is not None and language != "text":
                anchors = self._extract_anchor_lines(file_path, source, include_docstrings=include_docstrings)
                llm_attempts += 1
                try:
                    llm_summary, llm_symbols, ok = self._summarize_with_llm(
                        file_path=file_path,
                        language=language,
                        source=source,
                        anchors=anchors,
                    )
                    if ok and llm_summary.strip():
                        summary = llm_summary
                        merged_symbols = sorted(set(symbols) | set(llm_symbols))
                except Exception:
                    llm_failures += 1
                    logger.exception(
                        "LLM file summarization failed; falling back to local summary",
                        extra={"file_path": relative, "language": language},
                    )

            description = CodeDescription(
                file_path=relative,
                language=language,
                symbols=merged_symbols,
                summary=summary,
                entrypoint=entrypoint,
                symbol_docs=(symbol_docs if include_docstrings else {}),
            )
            descriptions.append(description)

            cache_updates[relative] = {
                "mtime": mtime,
                "language": language,
                "symbols": merged_symbols,
                "summary": summary,
                "entrypoint": entrypoint,
                "symbol_docs": description.symbol_docs,
            }

        if use_cache:
            self._save_cache(root, {"version": 2, "generated_at": datetime.now(timezone.utc).isoformat(), "files": cache_updates})

        if use_llm and self.llm_chain is not None:
            try:
                architecture_summary, tech_summary = self.llm_chain.synthesize_architecture(
                    dependency_report=dependency_report.to_dict(),
                    file_summaries=[item.to_dict() for item in descriptions],
                )
            except Exception:
                logger.exception("LLM architecture synthesis failed; falling back to local summary")
                architecture_summary = (
                    f"Repository contains {len(descriptions)} key files, "
                    f"{len(dependency_report.module_edges)} internal dependency edges, and "
                    f"{len(dependency_report.dependency_edges)} external dependency edges."
                )
                tech_summary = (
                    "Technologies: "
                    + ", ".join(dependency_report.technologies or dependency_report.frameworks or ["unknown"])
                )
        else:
            architecture_summary = (
                f"Repository contains {len(descriptions)} key files, "
                f"{len(dependency_report.module_edges)} internal dependency edges, and "
                f"{len(dependency_report.dependency_edges)} external dependency edges."
            )
            tech_summary = (
                "Technologies: "
                + ", ".join(dependency_report.technologies or dependency_report.frameworks or ["unknown"])
            )

        elapsed_ms = (perf_counter() - started) * 1000
        metrics = {
            "selected_files": len(selected),
            "cache_hits": cache_hits,
            "llm_attempts": llm_attempts,
            "llm_failures": llm_failures,
            "duration_ms": round(elapsed_ms, 3),
        }

        logger.info(
            "describe-completed",
            extra={
                "project_root": str(root),
                "selected_files": len(selected),
                "cache_hits": cache_hits,
                "llm_attempts": llm_attempts,
                "llm_failures": llm_failures,
                "duration_ms": round(elapsed_ms, 3),
            },
        )

        architecture_data = self._build_architecture_data(dependency_report, descriptions) if return_structured else {}
        architecture_mermaid = self._build_mermaid(dependency_report)

        return DescribeReport(
            project_root=str(root),
            selected_files=[str(item.relative_to(root)) for item in selected],
            descriptions=descriptions,
            architecture_summary=architecture_summary,
            tech_summary=tech_summary,
            chain_steps=chain_steps,
            architecture_mermaid=architecture_mermaid,
            architecture_data=architecture_data,
            metrics=metrics,
        )

    @staticmethod
    def render_markdown(report: DescribeReport) -> str:
        lines: list[str] = []
        lines.append("# Repository Description")
        lines.append("")
        lines.append(f"- Project root: `{report.project_root}`")
        lines.append(f"- Chain steps: {', '.join(report.chain_steps)}")
        if report.metrics:
            lines.append(f"- Metrics: `{json.dumps(report.metrics, sort_keys=True)}`")
        lines.append("")
        lines.append("## Architecture")
        lines.append(report.architecture_summary)
        lines.append("")
        if report.architecture_mermaid:
            lines.append("```mermaid")
            lines.append(report.architecture_mermaid)
            lines.append("```")
            lines.append("")
        lines.append("## Technology")
        lines.append(report.tech_summary)
        lines.append("")
        lines.append("## Files")
        for item in report.descriptions:
            marker = " [entrypoint]" if item.entrypoint else ""
            lines.append(f"- `{item.file_path}` ({item.language}){marker}")
            lines.append(f"  Summary: {item.summary}")
            lines.append(f"  Symbols: {', '.join(item.symbols) if item.symbols else 'none'}")
        return "\n".join(lines)

    @staticmethod
    def merge_llm_framework_hints(report: DependencyGraphReport, llm_frameworks: list[str]) -> DependencyGraphReport:
        if not llm_frameworks:
            return report
        merged_frameworks = sorted(set(report.frameworks) | set(llm_frameworks))
        merged_tech = sorted(set(report.technologies) | set(llm_frameworks))
        return DependencyGraphReport(
            project_root=report.project_root,
            package_managers=report.package_managers,
            frameworks=merged_frameworks,
            technologies=merged_tech,
            runtime_dependencies=report.runtime_dependencies,
            dev_dependencies=report.dev_dependencies,
            module_edges=report.module_edges,
            dependency_edges=report.dependency_edges,
            manifests=report.manifests,
            languages=report.languages,
        )
