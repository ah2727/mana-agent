from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path
from typing import Any
import traceback
from datetime import datetime
import typer
import asyncio
from asyncio.subprocess import PIPE
from rich.console import Console
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools.file_management import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from mana_analyzer.config.settings import (
    Settings,
    default_diagrams_dir,
    default_index_dir,
    default_llm_logs_dir,
    default_logs_dir,
    mana_root_dir,
)
from mana_analyzer.analysis.checks import PythonStaticAnalyzer
from mana_analyzer.analysis.chunker import CodeChunker
from mana_analyzer.parsers.multi_parser import MultiLanguageParser
from mana_analyzer.vector_store.faiss_store import FaissStore
from mana_analyzer.services.analyze_service import AnalyzeService
from mana_analyzer.services.ask_service import AskService
from mana_analyzer.describe.build import build_describe_service
from mana_analyzer.services.index_service import IndexService
from mana_analyzer.services.search_service import SearchService
from mana_analyzer.services.structure_service import StructureService
from mana_analyzer.services.dependency_service import DependencyService
from mana_analyzer.services.vulnerability_service import VulnerabilityService
from mana_analyzer.services.report_service import ReportService
# The deep‐flow LLM chain:
from mana_analyzer.describe.llm_chains.deep_flow import DeepFlowChain
from .output import build_output_sink, get_shared_console

logger = logging.getLogger(__name__)
console = get_shared_console()
app = typer.Typer(help="mana-analyzer CLI")

OUTPUT_DIR: Path | None = None



# ----------------------------------------------------------------------
# subprocess helper
# ----------------------------------------------------------------------

async def _spawn_command(cmd: list[str]) -> tuple[int, str, str]:
    """
    Spawn a subprocess, capture stdout/stderr, return (returncode, stdout, stderr).
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=PIPE,
        stderr=PIPE,
    )
    out_bytes, err_bytes = await proc.communicate()
    stdout = out_bytes.decode("utf-8", errors="ignore")
    stderr = err_bytes.decode("utf-8", errors="ignore")
    return proc.returncode, stdout, stderr
# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False, log_dir: Path | None = None) -> Path | None:
    log_level = logging.DEBUG if verbose else logging.INFO

    effective_log_dir = log_dir or default_logs_dir(Path.cwd())
    effective_log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = effective_log_dir / f"mana_analyzer_{timestamp}.log"

    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(file_handler)

    return log_file

def _resolve_output_file(
    output_dir: str | Path | None = None,
    *,
    default_name: str = "output",
    extension: str = "json",
    include_timestamp: bool = True,
) -> Path:
    # Backward-compatible: older command handlers pass positional `path` values.
    base_root = Path(output_dir) if output_dir is not None else (OUTPUT_DIR or Path.cwd())
    if base_root.suffix:
        base_root = base_root.parent
    base_dir = mana_root_dir(base_root)
    base_dir.mkdir(parents=True, exist_ok=True)

    if include_timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{default_name}_{ts}.{extension}"
    else:
        filename = f"{default_name}.{extension}"

    return base_dir / filename

def _resolve_analyze_artifact_paths(path: str | Path) -> tuple[Path, Path, Path]:
    root = Path(path).resolve()
    out_dir = root / ".mana"
    return out_dir / "analyze.json", out_dir / "analyze.md", out_dir / "analyze.html"


def _resolve_describe_artifact_paths(path: str | Path) -> tuple[Path, Path, Path]:
    root = Path(path).resolve()
    out_dir = mana_root_dir(root)
    return out_dir / "describe.json", out_dir / "describe.md", out_dir / "describe.html"

def _emit_json(payload, output_file: Path | None = None) -> None:
    sink = build_output_sink(command_name="cli", json_mode=True, output_file=output_file, console=console)
    sink.emit_json(payload)

def _emit_text(text: str, output_file: Path | None = None) -> None:
    sink = build_output_sink(command_name="cli", json_mode=False, output_file=output_file, console=console)
    sink.emit_text(text)

def _render_findings_markdown(findings) -> str:
    if not findings:
        return "## Findings\n\nNo findings.\n"

    lines = ["## Findings\n"]
    for f in findings:
        lines.append(
            f"- **{f.severity.upper()}** `{f.rule_id}` "
            f"{f.file_path}:{f.line}:{f.column} — {f.message}"
        )
    return "\n".join(lines) + "\n"

def _render_repository_summary_markdown(summary: dict) -> str:
    lines = ["## Repository Summary\n"]
    for k, v in summary.items():
        lines.append(f"- **{k}**: {v}")
    return "\n".join(lines) + "\n"

def _log_exception(context: str, exc: Exception, **extra) -> None:
    logger.error(
        f"Error during {context}: {exc}",
        extra={"context": context, **extra},
    )
    logger.debug(traceback.format_exc())

def _clamp_detail_line_target(detail_line_target: int) -> int:
    MIN_LINES = 100
    MAX_LINES = 2000

    if detail_line_target < MIN_LINES:
        return MIN_LINES
    if detail_line_target > MAX_LINES:
        return MAX_LINES
    return detail_line_target

def _resolve_report_artifact_paths(path: str) -> tuple[Path, Path, Path]:
    target = Path(path)
    if target.is_dir():
        default_json = target / "report.json"
        default_md   = target / "report.md"
        default_html = target / "report.html"
    else:
        base = target.with_suffix("") 
        default_json = base.with_suffix(".report.json")
        default_md   = base.with_suffix(".report.md")
        default_html = base.with_suffix(".report.html")
    return default_json, default_md, default_html

def _resolve_out_path(user_out: str | None, default: Path, suffix: str) -> Path:
    if user_out:
        p = Path(user_out)
        if p.is_dir():
            return p / default.name
        return p.with_suffix(suffix) if p.suffix == "" else p
    return default


# ---------------------------------------------------------------------------
# File agent (safe LangChain tools)
# ---------------------------------------------------------------------------

def build_file_agent(llm: Any, *, root_dir: Path | str = ".") -> Any:
    root = Path(root_dir).resolve()

    tools = [
        ReadFileTool(root_dir=str(root)),
        WriteFileTool(root_dir=str(root)),
        ListDirectoryTool(root_dir=str(root)),
    ]

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Vector store + search
# ---------------------------------------------------------------------------

def build_store(settings: Settings) -> FaissStore:
    embeddings = OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        model=settings.openai_embed_model,
        base_url=settings.openai_base_url,
    )
    return FaissStore(embeddings=embeddings)

def build_index_service(settings: Settings) -> IndexService:
    return IndexService(
        parser=MultiLanguageParser(),
        chunker=CodeChunker(),
        store=build_store(settings),
    )

def build_search_service(settings: Settings) -> SearchService:
    return SearchService(store=build_store(settings))


# ---------------------------------------------------------------------------
# Analyze services
# ---------------------------------------------------------------------------

def build_analyze_service() -> AnalyzeService:
    return AnalyzeService(analyzer=PythonStaticAnalyzer())


def build_llm_analyze_service(
    settings: Settings,
    model_override: str | None,
):
    model_name = model_override or settings.openai_chat_model

    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        model=model_name,
        base_url=settings.openai_base_url,
        temperature=0,
    )

    chain = AnalyzeChain(
        api_key=settings.openai_api_key,
        model=model_name,
        base_url=settings.openai_base_url,
    )

    file_agent = build_file_agent(llm)

    return chain, file_agent


# ---------------------------------------------------------------------------
# Ask service
# ---------------------------------------------------------------------------

def build_ask_service(
    settings: Settings,
    model_override: str | None,
    *,
    project_root: Path | None = None,
) -> AskService:
    model = model_override or settings.openai_chat_model
    root = project_root.resolve() if project_root else Path.cwd().resolve()

    qna_chain = QnAChain(
        api_key=settings.openai_api_key,
        model=model,
        base_url=settings.openai_base_url,
    )

    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        model=model,
        base_url=settings.openai_base_url,
        temperature=0,
    )

    file_agent = build_file_agent(llm, root_dir=root)

    ask_agent = AskAgent(
        api_key=settings.openai_api_key,
        model=model,
        base_url=settings.openai_base_url,
        project_root=root,
        file_agent=file_agent,
    )

    return AskService(qna_chain=qna_chain, ask_agent=ask_agent)


# ---------------------------------------------------------------------------
# Dependency service
# ---------------------------------------------------------------------------

def build_dependency_service() -> DependencyService:
    return DependencyService()


# ---------------------------------------------------------------------------
# Report service (with deep‐flow LLM injection)
# ---------------------------------------------------------------------------

def build_report_service(
    *,
    use_llm: bool,
    model_override: str | None,
    include_tests: bool,
) -> ReportService:
    dependency_service = build_dependency_service()
    analyze_service = build_analyze_service()
    
    settings = Settings()
    
    # Always build LLM components since LLM is now mandatory
    final_model = model_override or settings.openai_chat_model
    
    # 1) Build the DeepFlowChain for deep-profile flow synthesis
    llm_chain = DeepFlowChain(
        model_name=final_model,
        temperature=0,
        openai_api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    
    # 2) Build describe_service with LLM chain
    describe_service = build_describe_service(
        dependency_service=dependency_service,
        llm_chain=llm_chain,
        include_tests=include_tests,
    )
    
    structure_service = StructureService(include_tests=include_tests)
    vulnerability_service = VulnerabilityService()
    
    return ReportService(
        dependency_service=dependency_service,
        analyze_service=analyze_service,
        llm_analyze_service=None,
        describe_service=describe_service,
        structure_service=structure_service,
        vulnerability_service=vulnerability_service,
    )

# ---------------------------------------------------------------------------
# Export symbols
# ---------------------------------------------------------------------------

__all__ = [name for name in globals()]
