# mana_analyzer/describe/describe_service.py

from pathlib import Path
from typing import Any, Optional

from .file_summary_executor import FileSummaryExecutor
from mana_analyzer.dependencies.dependency_service import DependencyService
from mana_analyzer.utils.io import language_for_path


class DescribeService:
    def __init__(
        self,
        dependency_service: DependencyService,
        summary_executor: FileSummaryExecutor,
        llm_chain: Optional[Any] = None,
    ) -> None:
        """
        :param dependency_service: service that gathers files, imports, etc.
        :param summary_executor: service that summarizes each file
        :param llm_chain: an optional LLM chain implementing
                          `synthesize_deep_flow_analysis(...)`
        """
        self.dependency_service = dependency_service
        self.summary_executor = summary_executor
        self.llm_chain = llm_chain

    def describe(self, root: Path) -> dict:
        """
        Perform a basic (non-LLM) description of the project at `root`.
        Returns a dict with file summaries, symbols, imports, etc.
        """
        dependency_result = self.dependency_service.analyze(root)
        files = list(getattr(dependency_result, "files", []))

        # back-compat: some versions produce a language_map, others do not
        language_map = getattr(dependency_result, "language_map", None)
        if language_map is None:
            language_map = {path: language_for_path(path) for path in files}

        # back-compat: some versions produce a source_map, others not
        source_map = getattr(dependency_result, "source_map", None)
        if source_map is None:
            source_map = {}
            for path in files:
                try:
                    source_map[path] = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    source_map[path] = ""

        imports_map = getattr(dependency_result, "imports", None) or {}

        # Summarize each file
        summaries = self.summary_executor.summarize_files(
            files=files,
            language_map=language_map,
            source_map=source_map,
        )

        report_files = []
        for path in files:
            report_files.append({
                "path": str(path),
                "language": language_map.get(path),
                "summary": summaries[path]["summary"],
                "symbols": summaries[path]["symbols"],
                "imports": imports_map.get(path, []),
            })

        return {
            "root": str(root),
            "file_count": len(files),
            "files": report_files,
        }

    def synthesize_deep_flow_analysis(self, *args, **kwargs) -> Any:
        """
        Delegate to the injected llm_chain.  This must exist and
        provide a `synthesize_deep_flow_analysis(...)` method in order
        to run the --report-profile deep + --with-llm path.
        """
        if self.llm_chain is None or not hasattr(self.llm_chain, "synthesize_deep_flow_analysis"):
            raise RuntimeError(
                "LLM-only mode: describe_service.llm_chain.synthesize_deep_flow_analysis is required."
            )
        return self.llm_chain.synthesize_deep_flow_analysis(*args, **kwargs)
