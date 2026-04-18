# mana_analyzer/describe/build.py

from mana_analyzer.dependencies import DependencyService
from mana_analyzer.llm.repo_chain import RepositoryMultiChain

from .describe_service import DescribeService
from .file_summary_executor import FileSummaryExecutor
from typing import Any

def build_describe_service(
    dependency_service: DependencyService,
    llm_chain: Any | None = None,
    include_tests: bool = False,
) -> DescribeService:
    # Build the summary executor with the LLM chain
    summary_executor = FileSummaryExecutor(
        file_agent=None,       # not needed in LLM-only mode
        llm_chain=llm_chain,   # enables summarize_files_batch
    )

    # Build DescribeService with all three required parameters
    return DescribeService(
        dependency_service=dependency_service,
        summary_executor=summary_executor,
        llm_chain=llm_chain,   # enables synthesize_deep_flow_analysis
    )
