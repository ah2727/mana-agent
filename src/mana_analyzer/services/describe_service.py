from __future__ import annotations

import ast
import logging
from pathlib import Path

from mana_analyzer.analysis.models import CodeDescription, DependencyGraphReport, DescribeReport
from mana_analyzer.llm.repo_chain import RepositoryMultiChain
from mana_analyzer.services.dependency_service import DependencyService
from mana_analyzer.utils.io import iter_source_files

logger = logging.getLogger(__name__)


class DescribeService:
    def __init__(self, dependency_service: DependencyService, llm_chain: RepositoryMultiChain | None = None) -> None:
        self.dependency_service = dependency_service
        self.llm_chain = llm_chain

    @staticmethod
    def _language_from_suffix(path: Path) -> str:
        mapping = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".kt": "kotlin",
        }
        return mapping.get(path.suffix, "text")

    @staticmethod
    def _heuristic_symbols(path: Path, source: str, include_functions: bool) -> list[str]:
        if path.suffix != ".py":
            return []
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            return []
        symbols: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                symbols.append(node.name)
                if include_functions:
                    symbols.extend(
                        f"{node.name}.{item.name}"
                        for item in node.body
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    )
            elif include_functions and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(node.name)
        return sorted(set(symbols))

    @staticmethod
    def _select_files(root: Path, max_files: int) -> list[Path]:
        files = iter_source_files(root)
        scored: list[tuple[int, Path]] = []
        for file_path in files:
            name = file_path.name.lower()
            score = 0
            if name in {"readme.md", "pyproject.toml", "package.json"}:
                score += 8
            if "config" in name or "settings" in name:
                score += 5
            if file_path.suffix in {".py", ".ts", ".tsx", ".js"}:
                score += 3
            score += min(file_path.stat().st_size // 1024, 10)
            scored.append((score, file_path))
        scored.sort(key=lambda item: (-item[0], str(item[1])))
        return [item for _, item in scored[: max(1, max_files)]]

    @staticmethod
    def _local_summary(path: Path, source: str, language: str, symbols: list[str]) -> str:
        lines = source.splitlines()
        return (
            f"{path.name} is a {language} file with {len(lines)} lines and {len(symbols)} discovered symbols."
        )

    def describe(
        self,
        target_path: str | Path,
        max_files: int = 12,
        include_functions: bool = False,
        use_llm: bool = True,
    ) -> DescribeReport:
        root = Path(target_path).resolve()
        if root.is_file():
            root = root.parent

        chain_steps = [
            "file-tree-selection",
            "dependency-analysis",
            "file-summarization",
            "architecture-synthesis",
        ]

        selected = self._select_files(root, max_files=max_files)
        dependency_report = self.dependency_service.analyze(root)

        descriptions: list[CodeDescription] = []
        for file_path in selected:
            source = file_path.read_text(encoding="utf-8", errors="ignore")
            language = self._language_from_suffix(file_path)
            symbols = self._heuristic_symbols(file_path, source, include_functions=include_functions)
            if use_llm and self.llm_chain is not None and language != "text":
                summary, llm_symbols = self.llm_chain.summarize_file(
                    file_path=file_path,
                    language=language,
                    source=source[:12000],
                )
                merged_symbols = sorted(set(symbols) | set(llm_symbols))
            else:
                summary = self._local_summary(file_path, source, language, symbols)
                merged_symbols = symbols

            descriptions.append(
                CodeDescription(
                    file_path=str(file_path.relative_to(root)),
                    language=language,
                    symbols=merged_symbols,
                    summary=summary,
                )
            )

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

        return DescribeReport(
            project_root=str(root),
            selected_files=[str(item.relative_to(root)) for item in selected],
            descriptions=descriptions,
            architecture_summary=architecture_summary,
            tech_summary=tech_summary,
            chain_steps=chain_steps,
        )

    @staticmethod
    def render_markdown(report: DescribeReport) -> str:
        lines: list[str] = []
        lines.append("# Repository Description")
        lines.append("")
        lines.append(f"- Project root: `{report.project_root}`")
        lines.append(f"- Chain steps: {', '.join(report.chain_steps)}")
        lines.append("")
        lines.append("## Architecture")
        lines.append(report.architecture_summary)
        lines.append("")
        lines.append("## Technology")
        lines.append(report.tech_summary)
        lines.append("")
        lines.append("## Files")
        for item in report.descriptions:
            lines.append(f"- `{item.file_path}` ({item.language})")
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
