from __future__ import annotations

import asyncio
import ast
import json
import logging
import os
import re
import select
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import hashlib
from typing import Any

import typer
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table

from mana_analyzer.commands.ui_helpers import (
    ChatTurnTelemetry,
    RichToolCallbackHandler,
    UNLIMITED_AGENT_MAX_STEPS,
    _build_flow_summary_payload,
    _checkpoint_decisions_from_pass_window,
    _log_chat_turn,
    _pending_ui_selection_from_blocks,
    _read_chat_input,
    _render_answer_sections,
    _render_coding_sections,
    _render_dynamic_blocks,
    _render_flow_summary,
    _render_full_auto_checkpoint,
    _render_selection_block,
    _render_turn_transparency,
    _register_tool_if_missing,
    _resolve_payload_checklist_counts,
    _resolve_ui_selection_input,
    _run_with_live_buffer,
    _resolve_agent_max_steps,
    _log_call,
    _log_return,
    _EDIT_INTENT_TOKENS,
    _EDIT_TARGET_PATTERN,
)

from mana_analyzer.analysis.checks import PythonStaticAnalyzer
from mana_analyzer.analysis.chunker import CodeChunker
from mana_analyzer.config.settings import Settings, default_index_dir
from mana_analyzer.llm.ask_agent import AskAgent
from mana_analyzer.llm.analyze_chain import AnalyzeChain
from mana_analyzer.llm.qna_chain import QnAChain
from mana_analyzer.llm.repo_chain import RepositoryMultiChain
from mana_analyzer.llm.prompts import PLANNING_QUESTION_SYSTEM_PROMPT, PLANNING_SYSTEM_GUIDANCE
from mana_analyzer.llm.coding_agent import CodingAgent
from mana_analyzer.llm.run_logger import LlmRunLogger
from mana_analyzer.llm.tool_worker_process import ToolWorkerClient, ToolWorkerProcessError
from mana_analyzer.llm.tools_manager import ToolsManagerOrchestrator
from mana_analyzer.llm.tools_executor import (
    LocalToolsExecutor,
    RedisRQToolsExecutor,
    ToolsExecutionConfig,
)
from mana_analyzer.parsers.multi_parser import MultiLanguageParser
from mana_analyzer.services.analyze_service import AnalyzeService
from mana_analyzer.services.ask_service import AskService
from mana_analyzer.services.dependency_service import DependencyService
from mana_analyzer.services.describe_service import DescribeService
from mana_analyzer.services.index_service import IndexService
from mana_analyzer.services.llm_analyze_service import LlmAnalyzeService
from mana_analyzer.services.search_service import SearchService
from mana_analyzer.services.structure_service import StructureService
from mana_analyzer.services.coding_memory_service import CodingMemoryService
from mana_analyzer.tools import build_search_internet_tool
from mana_analyzer.utils.index_discovery import discover_index_dirs
from mana_analyzer.utils.logging import setup_logging
from mana_analyzer.utils.project_discovery import discover_subprojects
from mana_analyzer.vector_store.faiss_store import FaissStore
from mana_analyzer.services.chat_service import ChatService  # adjust import path when integrated

from mana_analyzer.services.vulnerability_service import VulnerabilityService
from mana_analyzer.services.report_service import ReportService
from langchain_core.callbacks.base import BaseCallbackHandler
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from rich.panel import Panel
from collections import deque
import time


app = typer.Typer(help="mana-analyzer CLI")
console = Console()
logger = logging.getLogger(__name__)
OUTPUT_DIR: Path | None = None
LLM_DEBUG_MODE: bool = False
UNLIMITED_AGENT_MAX_STEPS = 1_000_000_000


def _looks_like_edit_request(question: str) -> bool:
    lowered = (question or "").lower()
    if any(token in lowered for token in _EDIT_INTENT_TOKENS):
        return True
    has_edit_verb = bool(
        re.search(r"\b(update|edit|modify|patch|rewrite|refactor|fix|create|add|remove|delete|rename|document)\b", lowered)
    )
    return bool(has_edit_verb and _EDIT_TARGET_PATTERN.search(lowered))


def _looks_like_plan_trigger_request(question: str) -> bool:
    text = (question or "").strip().lower()
    if not text:
        return False
    return bool(re.search(r"\b(implement|execute|run|apply|trigger)\s+(the\s+|last\s+|that\s+|current\s+)?plan\b", text))


def _looks_like_plan_only_answer(answer_text: str, payload: dict | None = None) -> bool:
    text = (answer_text or "").strip().lower()
    if not text:
        text_empty = True
    else:
        text_empty = False
    if isinstance(payload, dict):
        ui_blocks = payload.get("ui_blocks")
        if isinstance(ui_blocks, list):
            has_plan = False
            has_non_plan = False
            for item in ui_blocks:
                if not isinstance(item, dict):
                    continue
                kind = str(item.get("type", "")).strip().lower()
                if kind == "plan":
                    has_plan = True
                elif kind:
                    has_non_plan = True
            # Auto-execute only when payload is truly plan-only.
            if has_plan and not has_non_plan and (
                text_empty or text.startswith("plan:") or text.startswith("execution plan:")
            ):
                return True
    if not text:
        return False
    if text.startswith("plan:"):
        return True
    if text.startswith("execution plan:"):
        return True
    return ("plan:" in text and "summary" not in text and "next step" not in text)


def _sanitize_full_auto_answer_text(
    answer_text: str,
    *,
    changed_files_count: int = 0,
    terminal_reason: str = "",
) -> str:
    def _normalize_terminal_reason(reason: str) -> str:
        return str(reason or "").strip().lower()

    def _looks_like_hard_blocker_text(text: str) -> bool:
        lowered_text = str(text or "").strip().lower()
        if not lowered_text:
            return False
        patterns = (
            "missing credential",
            "missing credentials",
            "missing api key",
            "missing token",
            "missing secret",
            "permission denied",
            "insufficient permission",
            "unauthorized",
            "forbidden",
            "access denied",
            "missing target identifier",
            "target identifier required",
            "missing identifier",
            "identifier is required",
            "missing file path",
            "path is required",
            "target path required",
            "provide file path",
            "unavailable",
        )
        return any(token in lowered_text for token in patterns)

    def _looks_like_non_hard_blocker_text(text: str) -> bool:
        lowered_text = str(text or "").strip().lower()
        if not lowered_text:
            return False
        if _looks_like_hard_blocker_text(lowered_text):
            return False
        patterns = (
            "blocker decision",
            "need one blocker decision",
            "scope choice",
            "scope decision",
            "need scope",
            "which scope",
            "choose scope",
            "choose option",
            "which option",
            "option 1",
            "option 2",
            "1 or 2",
            "one or two",
            "pick one",
            "awaiting scope decision",
            "awaiting your decision",
            "i'm blocked on",
            "i am blocked on",
            "blocked on making",
            "need to read",
            "need to inspect",
            "need to review",
            "before patching",
            "before editing",
            "before making changes",
            "share permission to proceed",
            "permission to proceed",
            "requires explicit tool execution",
            "tool access is available",
            "once tool access is available",
        )
        return any(token in lowered_text for token in patterns)

    text = str(answer_text or "").strip()
    if not text:
        reason = str(terminal_reason or "").strip()
        reason_key = _normalize_terminal_reason(reason)
        if reason_key == "pass_cap_reached":
            return "Status: executing."
        if reason:
            if any(token in reason for token in ("blocked", "error", "unavailable", "violation")):
                return f"Status: blocked ({reason})."
            return f"Status: executing ({reason})."
        return "Status: executing."

    lowered = text.lower()
    confirmation_patterns = (
        "if you want",
        "reply \"yes",
        "reply 'yes",
        "say yes",
        "type yes",
        "let me know if you want",
    )
    has_confirmation_prompt = any(token in lowered for token in confirmation_patterns)
    has_non_hard_blocker_prompt = _looks_like_non_hard_blocker_text(lowered)
    if has_confirmation_prompt or has_non_hard_blocker_prompt:
        reason = str(terminal_reason or "").strip()
        reason_key = _normalize_terminal_reason(reason)
        reason_and_text = f"{reason}\n{text}"
        if changed_files_count > 0:
            return "Status: completed."
        if reason_key == "pass_cap_reached":
            return "Status: executing."
        if reason and (
            any(token in reason for token in ("blocked", "error", "unavailable", "violation"))
            or _looks_like_hard_blocker_text(reason_and_text)
        ):
            return f"Status: blocked ({reason})."
        if reason:
            return f"Status: executing ({reason})."
        return "Status: executing."
    return text


def _auto_select_ui_option(block: dict[str, Any]) -> dict[str, Any] | None:
    options = block.get("options")
    if not isinstance(options, list):
        return None
    normalized: list[dict[str, Any]] = [item for item in options if isinstance(item, dict)]
    if not normalized:
        return None
    return normalized[0]


def _render_auto_execute_pass_status(
    console: Console,
    *,
    objective: str,
    pass_index: int,
    pass_cap: int,
    planner_step_id: str = "",
    planner_step_title: str = "",
    planner_decision: str = "",
    planner_decision_reason: str = "",
    batch_reason: str = "",
    expected_progress: str,
) -> None:
    console.print("\n[bold cyan]Auto-Execute[/bold cyan]")
    console.print(f"- objective: {objective or '-'}")
    console.print(f"- pass: {pass_index}/{pass_cap}")
    if planner_step_id or planner_step_title:
        step = planner_step_id or "-"
        if planner_step_title:
            step = f"{step} ({planner_step_title})" if planner_step_id else planner_step_title
        console.print(f"- step: {step}")
    if planner_decision:
        console.print(f"- decision: {planner_decision}")
    if planner_decision_reason:
        console.print(f"- reason: {planner_decision_reason}")
    if batch_reason:
        console.print(f"- batch reason: {batch_reason}")
    console.print(f"- expected progress: {expected_progress or '-'}")


def _render_prechecklist_preview(
    console: Console,
    *,
    prechecklist: dict[str, Any],
    warning: str = "",
) -> None:
    objective = str(prechecklist.get("objective", "") or "").strip()
    source = str(prechecklist.get("source", "") or "").strip()
    steps = prechecklist.get("steps") if isinstance(prechecklist.get("steps"), list) else []
    console.print("[cyan]Executing active flow plan...[/cyan]")
    if objective:
        console.print(f"[bold]Plan objective:[/bold] {objective}")
    if source:
        console.print(f"[dim]Checklist source: {source}[/dim]")
    if steps:
        console.print("[bold]Checklist[/bold]")
        for item in steps[:12]:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status", "pending") or "pending")
            title = str(item.get("title", "step") or "step")
            console.print(f"- [{status}] {title}")
    if warning.strip():
        console.print(Panel(f"- {warning.strip()}", title="Planner Warning", border_style="yellow"))


def _planning_questions(max_questions: int) -> list[str]:
    ordered = [
        "What is the concrete goal and the success criteria?",
        "What is in scope vs out of scope, and what hard constraints must we honor?",
        "What output format, depth, and key tradeoffs should the plan optimize for?",
        "What rollout, migration, or compatibility constraints should be included?",
        "What tests or acceptance criteria are mandatory before completion?",
        "What risks should be prioritized and explicitly mitigated?",
    ]
    limit = max(1, min(max_questions, len(ordered)))
    return ordered[:limit]


def _build_planning_question(user_request: str, asked_count: int, max_questions: int) -> str:
    prompts = _planning_questions(max_questions)
    question = prompts[min(asked_count, len(prompts) - 1)]
    return (
        f"[cyan]Planning request:[/cyan] {user_request}\n"
        f"[bold]Planning question {asked_count + 1}/{len(prompts)}[/bold]\n"
        f"{question}"
    )


def _normalize_planning_question_text(text: str, fallback: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return fallback
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            raw = "\n".join(lines[1:-1]).strip()
    lines = [line.strip("-* ").strip() for line in raw.splitlines() if line.strip()]
    candidate = lines[0] if lines else raw
    candidate = re.sub(r"^planning question\s*\d+\s*[:.-]\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"^\d+\s*[\).\:-]\s*", "", candidate)
    candidate = candidate.strip().strip('"').strip("'")
    if not candidate:
        return fallback
    if len(candidate) > 220:
        candidate = candidate[:220].rstrip()
    if not candidate.endswith("?"):
        candidate = candidate.rstrip(".!") + "?"
    return candidate


def _generate_planning_question_llm(
    *,
    ask_service: AskService,
    planning_request: str,
    prior_questions: list[str],
    prior_answers: list[str],
    asked_count: int,
    max_questions: int,
) -> str:
    fallback = _planning_questions(max_questions)[min(asked_count, max_questions - 1)]
    qna_chain = getattr(ask_service, "qna_chain", None)
    llm = getattr(qna_chain, "llm", None)
    if llm is None or not hasattr(llm, "invoke"):
        raise RuntimeError("planning question LLM unavailable")

    qa_lines: list[str] = []
    for idx, (question, answer) in enumerate(zip(prior_questions, prior_answers), start=1):
        qa_lines.append(f"- Q{idx}: {question}")
        qa_lines.append(f"  A{idx}: {answer}")
    qa_block = "\n".join(qa_lines) if qa_lines else "- none yet"

    human_prompt = (
        f"User planning request:\n{planning_request}\n\n"
        f"Current question index: {asked_count + 1}/{max_questions}\n\n"
        f"Previously asked planning questions:\n"
        + ("\n".join(f"- {q}" for q in prior_questions) if prior_questions else "- none")
        + "\n\n"
        f"Clarification answers so far:\n{qa_block}\n\n"
        "Return exactly one best next clarification question."
    )
    response = llm.invoke(
        [
            SystemMessage(content=PLANNING_QUESTION_SYSTEM_PROMPT),
            HumanMessage(content=human_prompt),
        ]
    )
    return _normalize_planning_question_text(str(getattr(response, "content", "") or ""), fallback)


def _build_planning_instruction(
    user_request: str,
    answers: list[str],
    max_questions: int,
    questions: list[str] | None = None,
) -> str:
    prompts = list(questions or _planning_questions(max_questions))
    qa_lines: list[str] = []
    for idx, answer in enumerate(answers):
        question = prompts[min(idx, len(prompts) - 1)]
        qa_lines.append(f"- Q{idx+1}: {question}")
        qa_lines.append(f"  A{idx+1}: {answer}")
    qa_block = "\n".join(qa_lines) if qa_lines else "- (no clarifications provided)"
    return (
        f"{PLANNING_SYSTEM_GUIDANCE}\n\n"
        "User request:\n"
        f"{user_request}\n\n"
        "Clarifications:\n"
        f"{qa_block}\n\n"
        "Return a complete, implementation-ready plan in Markdown."
    )


def _index_has_vectors(index_dir: Path) -> bool:
    _log_call("_index_has_vectors", index_dir=str(index_dir))
    faiss_dir = index_dir / "faiss"
    if not faiss_dir.exists() or not faiss_dir.is_dir():
        _log_return("_index_has_vectors", result=False, reason="missing_or_not_dir", faiss_dir=str(faiss_dir))
        return False
    result = any(faiss_dir.iterdir())
    _log_return("_index_has_vectors", result=result, faiss_dir=str(faiss_dir))
    return result


def _index_has_chunks(index_dir: Path) -> bool:
    _log_call("_index_has_chunks", index_dir=str(index_dir))
    chunks_file = index_dir / "chunks.jsonl"
    result = chunks_file.exists() and chunks_file.stat().st_size > 0
    _log_return(
        "_index_has_chunks",
        result=result,
        chunks_file=str(chunks_file),
        exists=chunks_file.exists(),
        size=(chunks_file.stat().st_size if chunks_file.exists() else 0),
    )
    return result


def _index_has_search_data(index_dir: Path) -> bool:
    _log_call("_index_has_search_data", index_dir=str(index_dir))
    result = _index_has_vectors(index_dir) or _index_has_chunks(index_dir)
    _log_return("_index_has_search_data", result=result)
    return result


# ----------------------------
# Ephemeral index helpers
# ----------------------------


def _make_ephemeral_index_dir(prefix: str = "mana_index_") -> tuple[tempfile.TemporaryDirectory, Path]:
    tmp = tempfile.TemporaryDirectory(prefix=prefix)
    return tmp, Path(tmp.name).resolve()


def _stable_subdir_name(path: Path) -> str:
    h = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{path.name}-{h}"


# ----------------------------
# Report artifact helpers
# ----------------------------


def _resolve_report_artifact_dir(project_root: Path) -> Path:
    _log_call(
        "_resolve_report_artifact_dir",
        project_root=str(project_root),
        output_dir=str(OUTPUT_DIR) if OUTPUT_DIR else None,
    )
    if OUTPUT_DIR is not None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _log_return("_resolve_report_artifact_dir", resolved=str(OUTPUT_DIR))
        return OUTPUT_DIR
    target = project_root / ".mana_logs"
    target.mkdir(parents=True, exist_ok=True)
    _log_return("_resolve_report_artifact_dir", resolved=str(target))
    return target


def _resolve_report_artifact_paths(target_path: str | Path) -> tuple[Path, Path]:
    _log_call("_resolve_report_artifact_paths", target_path=str(target_path))
    target = Path(target_path).resolve()
    project_root = target if target.is_dir() else target.parent
    stamp = datetime.now().strftime("%Y%m%d-%H")
    stem = f"{project_root.name}-{stamp}-report"
    out_dir = _resolve_report_artifact_dir(project_root)
    json_path, md_path = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    _log_return("_resolve_report_artifact_paths", json=str(json_path), markdown=str(md_path))
    return json_path, md_path


def _resolve_analyze_artifact_paths(target_path: str | Path) -> tuple[Path, Path]:
    _log_call("_resolve_analyze_artifact_paths", target_path=str(target_path))
    target = Path(target_path).resolve()
    project_root = target if target.is_dir() else target.parent
    stamp = datetime.now().strftime("%Y%m%d-%H")
    stem = f"{project_root.name}-{stamp}-analyze"
    out_dir = _resolve_report_artifact_dir(project_root)
    json_path, md_path = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    _log_return("_resolve_analyze_artifact_paths", json=str(json_path), markdown=str(md_path))
    return json_path, md_path


def _resolve_out_path(arg: str | None, default_path: Path, *, suffix: str) -> Path:
    _log_call("_resolve_out_path", arg=arg, default=str(default_path), suffix=suffix)
    if not arg:
        _log_return("_resolve_out_path", resolved=str(default_path), mode="default")
        return default_path
    p = Path(arg).expanduser().resolve()
    if p.exists() and p.is_dir():
        resolved = p / default_path.name
        _log_return("_resolve_out_path", resolved=str(resolved), mode="existing_dir")
        return resolved
    if str(arg).endswith(("/", "\\")):
        p.mkdir(parents=True, exist_ok=True)
        resolved = p / default_path.name
        _log_return("_resolve_out_path", resolved=str(resolved), mode="dir_intent_created")
        return resolved
    if p.suffix == "":
        resolved = p.with_suffix(suffix)
        _log_return("_resolve_out_path", resolved=str(resolved), mode="added_suffix")
        return resolved
    _log_return("_resolve_out_path", resolved=str(p), mode="as_is")
    return p


def _clamp_detail_line_target(value: int) -> int:
    _log_call("_clamp_detail_line_target", value=value)
    if value < 300:
        _log_return("_clamp_detail_line_target", result=300, clamped=True)
        return 300
    if value > 400:
        _log_return("_clamp_detail_line_target", result=400, clamped=True)
        return 400
    _log_return("_clamp_detail_line_target", result=value, clamped=False)
    return value


def _render_repository_summary_markdown(summary_payload: dict) -> str:
    descriptions = summary_payload.get("descriptions") or []
    selected_files = summary_payload.get("selected_files") or []
    architecture_summary = str(summary_payload.get("architecture_summary") or "").strip()
    tech_summary = str(summary_payload.get("tech_summary") or "").strip()

    lines = [
        "## Repository Summary",
        f"- Selected files: {len(selected_files)}",
        f"- Summarized files: {len(descriptions)}",
    ]

    if architecture_summary:
        lines.extend(["", "### Architecture Summary", architecture_summary])
    if tech_summary:
        lines.extend(["", "### Technology Summary", tech_summary])

    if descriptions:
        lines.extend(["", "### File Summaries"])
        for item in descriptions[:12]:
            file_path = item.get("file_path", "unknown")
            language = item.get("language", "unknown")
            summary = str(item.get("summary") or "").strip()
            if summary:
                lines.append(f"- `{file_path}` ({language}): {summary}")
            else:
                lines.append(f"- `{file_path}` ({language})")

    return "\n".join(lines).rstrip()


def _render_findings_markdown(findings: list[Finding]) -> str:
    lines = ["# Analyze Findings", ""]
    if not findings:
        lines.append("No findings.")
        return "\n".join(lines).rstrip()

    lines.append(f"- Total findings: {len(findings)}")
    lines.append("")
    lines.append("## Findings")
    for finding in findings:
        lines.append(
            f"- [{finding.severity}] `{finding.rule_id}` `{finding.file_path}:{finding.line}:{finding.column}` {finding.message}"
        )
    return "\n".join(lines).rstrip()


def build_store(settings: Settings) -> FaissStore:
    _log_call(
        "build_store",
        embed_model=getattr(settings, "openai_embed_model", None),
        has_base_url=bool(getattr(settings, "openai_base_url", None)),
    )
    kwargs = {"api_key": settings.openai_api_key, "model": settings.openai_embed_model}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    embeddings = OpenAIEmbeddings(**kwargs)
    store = FaissStore(embeddings=embeddings)
    _log_return("build_store", store_type=type(store).__name__)
    return store


def build_index_service(settings: Settings) -> IndexService:
    _log_call("build_index_service")
    svc = IndexService(parser=MultiLanguageParser(), chunker=CodeChunker(), store=build_store(settings))
    _log_return("build_index_service", service_type=type(svc).__name__)
    return svc


def _index_service_index_compat(
    index_service: IndexService,
    *,
    target_path: Path,
    index_dir: Path,
    rebuild: bool,
    vectors: bool | None = None,
) -> object:
    try:
        if vectors is None:
            return index_service.index(target_path=target_path, index_dir=index_dir, rebuild=rebuild)
        return index_service.index(target_path=target_path, index_dir=index_dir, rebuild=rebuild, vectors=vectors)
    except TypeError as exc:
        if "vectors" not in str(exc):
            raise
        return index_service.index(target_path=target_path, index_dir=index_dir, rebuild=rebuild)


def build_search_service(settings: Settings) -> SearchService:
    _log_call("build_search_service")
    svc = SearchService(store=build_store(settings))
    _log_return("build_search_service", service_type=type(svc).__name__)
    return svc


def build_analyze_service() -> AnalyzeService:
    _log_call("build_analyze_service")
    svc = AnalyzeService(analyzer=PythonStaticAnalyzer())
    _log_return("build_analyze_service", service_type=type(svc).__name__)
    return svc


def build_llm_analyze_service(settings: Settings, model_override: str | None) -> LlmAnalyzeService:
    model = model_override or settings.openai_chat_model
    _log_call("build_llm_analyze_service", model=model, model_override=model_override)
    chain = AnalyzeChain(
        api_key=settings.openai_api_key,
        model=model,
        base_url=settings.openai_base_url,
    )
    svc = LlmAnalyzeService(analyze_chain=chain)
    _log_return("build_llm_analyze_service", service_type=type(svc).__name__)
    return svc


def build_ask_service(
    settings: Settings,
    model_override: str | None,
    *,
    project_root: Path | None = None,
) -> AskService:
    model = model_override or settings.openai_chat_model
    root = project_root.resolve() if project_root else Path.cwd().resolve()

    _log_call("build_ask_service", model=model, model_override=model_override, project_root=str(root))

    qna = QnAChain(
        api_key=settings.openai_api_key,
        model=model,
        base_url=settings.openai_base_url,
    )

    search_service = build_search_service(settings)

    agent = AskAgent(
        api_key=settings.openai_api_key,
        model=model,
        base_url=settings.openai_base_url,
        search_service=search_service,
        project_root=root,
    )
    _register_tool_if_missing(agent, build_search_internet_tool())

    svc = AskService(
        store=build_store(settings),
        qna_chain=qna,
        ask_agent=agent,
        search_service=search_service,
    )
    _log_return("build_ask_service", service_type=type(svc).__name__)
    return svc


def _build_ask_service_compat(
    settings: Settings,
    model_override: str | None,
    *,
    project_root: Path | None = None,
) -> AskService:
    if project_root is None:
        return build_ask_service(settings, model_override=model_override)
    try:
        return build_ask_service(settings, model_override=model_override, project_root=project_root)
    except TypeError as exc:
        if "project_root" not in str(exc):
            raise
        return build_ask_service(settings, model_override=model_override)


def build_dependency_service() -> DependencyService:
    _log_call("build_dependency_service")
    svc = DependencyService()
    _log_return("build_dependency_service", service_type=type(svc).__name__)
    return svc


def build_repo_chain(settings: Settings, model_override: str | None) -> RepositoryMultiChain:
    model = model_override or settings.openai_chat_model
    _log_call("build_repo_chain", model=model, model_override=model_override)
    chain = RepositoryMultiChain(
        api_key=settings.openai_api_key,
        model=model,
        base_url=settings.openai_base_url,
    )
    _log_return("build_repo_chain", chain_type=type(chain).__name__)
    return chain


def build_describe_service(settings: Settings, model_override: str | None, use_llm: bool) -> DescribeService:
    _log_call("build_describe_service", use_llm=use_llm, model_override=model_override)
    llm_chain = build_repo_chain(settings, model_override) if use_llm else None
    svc = DescribeService(dependency_service=build_dependency_service(), llm_chain=llm_chain)
    _log_return("build_describe_service", service_type=type(svc).__name__)
    return svc


def build_report_service(
    *,
    use_llm: bool,
    model_override: str | None,
    include_tests: bool,
) -> ReportService:
    _log_call("build_report_service", use_llm=use_llm, model_override=model_override, include_tests=include_tests)
    settings = Settings() if use_llm else None

    dependency_service = build_dependency_service()
    analyze_service = build_analyze_service()

    llm_analyze_service = None
    if use_llm:
        assert settings is not None
        llm_analyze_service = build_llm_analyze_service(settings, model_override=model_override)
        describe_service = build_describe_service(settings, model_override=model_override, use_llm=True)
    else:
        describe_service = DescribeService(dependency_service=dependency_service, llm_chain=None)

    structure_service = StructureService(include_tests=include_tests)
    vuln_service = VulnerabilityService()

    svc = ReportService(
        dependency_service=dependency_service,
        analyze_service=analyze_service,
        llm_analyze_service=llm_analyze_service,
        describe_service=describe_service,
        structure_service=structure_service,
        vulnerability_service=vuln_service,
    )
    _log_return("build_report_service", service_type=type(svc).__name__)
    return svc


def _resolve_output_file(target_path: str | Path | None = None) -> Path | None:
    _log_call(
        "_resolve_output_file",
        target_path=str(target_path) if target_path else None,
        output_dir=str(OUTPUT_DIR) if OUTPUT_DIR else None,
    )
    if OUTPUT_DIR is None:
        _log_return("_resolve_output_file", result=None, reason="OUTPUT_DIR_none")
        return None
    target = Path(target_path).resolve() if target_path else Path.cwd().resolve()
    root = target if target.is_dir() else target.parent
    stamp = datetime.now().strftime("%Y%m%d-%H")
    file_name = f"{root.name}-{stamp}.log"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / file_name
    _log_return("_resolve_output_file", result=str(out))
    return out


def _append_output(output_file: Path | None, content: str) -> None:
    _log_call("_append_output", output_file=str(output_file) if output_file else None, content_len=len(content or ""))
    if output_file is None or not content:
        _log_return("_append_output", skipped=True)
        return
    try:
        with output_file.open("a", encoding="utf-8") as handle:
            handle.write(content.rstrip() + "\n")
        _log_return("_append_output", wrote=True, path=str(output_file))
    except Exception as exc:
        _log_exception("_append_output", exc, path=str(output_file))


def _emit_text(content: str, output_file: Path | None) -> None:
    _log_call("_emit_text", content_len=len(content or ""), output_file=str(output_file) if output_file else None)
    console.print(content)
    _append_output(output_file, content)
    _log_return("_emit_text", ok=True)


def _emit_json(payload: object, output_file: Path | None) -> None:
    _log_call("_emit_json", output_file=str(output_file) if output_file else None)
    text = json.dumps(payload)
    console.print_json(text)
    _append_output(output_file, json.dumps(payload, indent=2))
    _log_return("_emit_json", ok=True)


async def _spawn_command(cmd: list[str]) -> tuple[int, str, str]:
    _log_call("_spawn_command", cmd=cmd)
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        text=False,
    )
    stdout, stderr = await process.communicate()
    decode = lambda value: value.decode("utf-8", errors="ignore").strip()
    rc, out, err = process.returncode, decode(stdout), decode(stderr)
    _log_return("_spawn_command", returncode=rc, stdout_len=len(out), stderr_len=len(err))
    return rc, out, err


def _truncate_output(text: str, limit: int = 4000) -> str:
    _log_call("_truncate_output", text_len=len(text or ""), limit=limit)
    if len(text) <= limit:
        _log_return("_truncate_output", truncated=False, result_len=len(text))
        return text
    result = text[:limit] + "\n[truncated]"
    _log_return("_truncate_output", truncated=True, result_len=len(result))
    return result


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logs."),
    debug_llm: bool = typer.Option(
        False,
        "--debug-llm/--no-debug-llm",
        help="Show internal LLM transport/request logs in live chat panels.",
    ),
    log_dir: str | None = typer.Option(None, "--log-dir", help="Directory for application log files."),
    output_dir: str | None = typer.Option(None, "--output-dir", help="Directory for saving command output logs."),
) -> None:
    global OUTPUT_DIR, LLM_DEBUG_MODE
    OUTPUT_DIR = Path(output_dir).resolve() if output_dir else None
    LLM_DEBUG_MODE = debug_llm
    log_file = setup_logging(verbose=verbose, log_dir=log_dir)
    if not debug_llm:
        for noisy_logger in ("openai", "httpx", "httpcore"):
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    logger.debug(
        "CLI initialized",
        extra={
            "verbose": verbose,
            "debug_llm": debug_llm,
            "log_file": str(log_file),
            "output_dir": str(OUTPUT_DIR) if OUTPUT_DIR else None,
        },
    )
    if tuple(sys.version_info[:2]) >= (3, 14):
        warning_msg = (
            "Python 3.14+ may emit LangChain/Pydantic v1 compatibility warnings. "
            "Recommended runtime: Python 3.12 or 3.13."
        )
        warnings.warn(warning_msg, UserWarning, stacklevel=2)
        console.print(f"[yellow]Warning:[/yellow] {warning_msg}")



@app.command()
def index(
    path: str,
    index_dir: str | None = typer.Option(None, "--index-dir"),
    rebuild: bool = typer.Option(False, "--rebuild"),
    ephemeral_index: bool = typer.Option(
        False,
        "--ephemeral-index",
        help="Use a temporary index dir and delete it after the command (ignored if --index-dir is set).",
    ),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    output_file = _resolve_output_file(path)
    logger.info("Index command started", extra={"path": path, "rebuild": rebuild, "index_dir": index_dir, "ephemeral_index": ephemeral_index})
    settings = Settings()
    service = build_index_service(settings)

    tmp: tempfile.TemporaryDirectory | None = None
    if ephemeral_index and not index_dir:
        tmp, resolved_index_dir = _make_ephemeral_index_dir()
    else:
        resolved_index_dir = Path(index_dir).resolve() if index_dir else default_index_dir(path)

    logger.debug("Resolved index directory", extra={"index_dir": str(resolved_index_dir)})

    try:
        result = service.index(target_path=path, index_dir=resolved_index_dir, rebuild=rebuild)
        logger.info("Index command completed", extra={**result, "path": path, "index_dir": str(resolved_index_dir)})
    except Exception as exc:
        _log_exception("index_command", exc, path=path, index_dir=str(resolved_index_dir))
        raise
    finally:
        if tmp is not None:
            tmp.cleanup()

    if as_json:
        _emit_json(result, output_file=output_file)
    else:
        rendered = "\n".join(f"{key}: {value}" for key, value in result.items())
        _emit_text(rendered, output_file=output_file)


@app.command()
def search(
    query: str,
    k: int | None = typer.Option(None, "--k"),
    index_dir: str | None = typer.Option(None, "--index-dir"),
    ephemeral_index: bool = typer.Option(
        False,
        "--ephemeral-index",
        help="Build a temporary index of the current directory, search it, then delete it (ignored if --index-dir is set).",
    ),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    output_file = _resolve_output_file()
    logger.info("Search command started", extra={"query": query, "k": k, "index_dir": index_dir, "ephemeral_index": ephemeral_index})
    settings = Settings()
    resolved_k = k or settings.default_top_k

    tmp: tempfile.TemporaryDirectory | None = None
    if ephemeral_index and not index_dir:
        tmp, resolved_index_dir = _make_ephemeral_index_dir()
        # Ensure index exists for search
        index_service = build_index_service(settings)
        _index_service_index_compat(
            index_service,
            target_path=Path.cwd(),
            index_dir=resolved_index_dir,
            rebuild=False,
            vectors=True,
        )
    else:
        resolved_index_dir = Path(index_dir).resolve() if index_dir else default_index_dir(Path.cwd())

    logger.debug("Resolved search parameters", extra={"k": resolved_k, "index_dir": str(resolved_index_dir)})

    service = build_search_service(settings)

    try:
        results = service.search(index_dir=resolved_index_dir, query=query, k=resolved_k)
        logger.info("Search command completed", extra={"hits": len(results), "query": query})
    except Exception as exc:
        _log_exception("search_command", exc, query=query, index_dir=str(resolved_index_dir), k=resolved_k)
        raise
    finally:
        if tmp is not None:
            tmp.cleanup()

    if as_json:
        _emit_json([item.to_dict() for item in results], output_file=output_file)
        return

    if not results:
        _emit_text("No results found.", output_file=output_file)
        return

    lines: list[str] = []
    for idx, hit in enumerate(results, start=1):
        lines.append(f"[{idx}] {hit.symbol_name} ({hit.score:.3f})")
        lines.append(f"  {hit.file_path}:{hit.start_line}-{hit.end_line}")
        lines.append(f"  {hit.snippet[:180].strip()}")
        lines.append("")
    _emit_text("\n".join(lines).rstrip(), output_file=output_file)


@app.command("flow")
def flow_cmd(
    project_path: str | None = typer.Argument(
        None,
        help="Project root containing .mana_index/chat_memory.sqlite3.",
    ),
    flow_id: str | None = typer.Option(None, "--flow-id", help="Flow ID to inspect; defaults to active flow."),
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
    max_turns: int = typer.Option(5, "--max-turns", help="Maximum recent turns to include."),
    max_tasks: int = typer.Option(20, "--max-tasks", help="Maximum open tasks to include."),
) -> None:
    resolved_project_path = project_path or "."
    output_file = _resolve_output_file(resolved_project_path)
    project_root = Path(resolved_project_path).resolve()
    resolved_format = str(output_format or "text").strip().lower()
    if resolved_format not in {"text", "json"}:
        raise typer.BadParameter("--format must be 'text' or 'json'.")

    memory_service = CodingMemoryService(
        project_root=project_root,
        max_turns=max(1, int(max_turns)),
        max_tasks=max(1, int(max_tasks)),
    )

    target_flow_id = flow_id or memory_service.get_active_flow_id()
    if not target_flow_id:
        _emit_text("No active coding flow found.", output_file=output_file)
        return

    summary_payload = _build_flow_summary_payload(memory_service, str(target_flow_id))
    if summary_payload is None:
        _emit_text(f"Flow not found: {target_flow_id}", output_file=output_file)
        return

    if resolved_format == "json":
        _emit_json(summary_payload, output_file=output_file)
        return

    capture_console = Console(record=True)
    _render_flow_summary(
        capture_console,
        summary_payload,
        include_checklist=True,
        include_transitions=True,
        include_recent_turns=True,
    )
    _emit_text(capture_console.export_text().rstrip(), output_file=output_file)


@app.command()
def analyze(
    path: str,
    fail_on: str = typer.Option("none", "--fail-on"),
    with_llm: bool = typer.Option(False, "--with-llm"),
    model: str | None = typer.Option(None, "--model"),
    llm_max_files: int = typer.Option(10, "--llm-max-files"),
    full_structure: bool = typer.Option(False, "--full-structure"),
    tech_summary: bool = typer.Option(False, "--tech-summary"),
    include_tests: bool = typer.Option(False, "--include-tests"),
    output_format: str = typer.Option("both", "--output-format"),
    chain_profile: str = typer.Option("default", "--chain-profile"),
    chain_config: str | None = typer.Option(None, "--chain-config"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    output_file = _resolve_output_file(path)
    logger.info(
        "Analyze command started",
        extra={
            "path": path,
            "fail_on": fail_on,
            "with_llm": with_llm,
            "model_override": model,
            "llm_max_files": llm_max_files,
            "full_structure": full_structure,
            "tech_summary": tech_summary,
            "include_tests": include_tests,
            "output_format": output_format,
            "chain_profile": chain_profile,
            "chain_config": chain_config,
        },
    )
    settings = Settings()
    service = build_analyze_service()

    try:
        static_findings = service.analyze(path)
        logger.debug("Static analyze completed", extra={"count": len(static_findings), "path": path})
    except Exception as exc:
        _log_exception("analyze_static", exc, path=path)
        raise

    llm_findings = []
    if with_llm:
        logger.info("LLM analyze mode enabled")
        try:
            llm_service = build_llm_analyze_service(settings, model_override=model)
            llm_findings = llm_service.analyze(path, static_findings=static_findings, max_files=llm_max_files)
            logger.debug("LLM analyze completed", extra={"count": len(llm_findings), "path": path})
        except Exception as exc:
            _log_exception("analyze_llm", exc, path=path, llm_max_files=llm_max_files, model_override=model)
            raise
    else:
        logger.info("LLM analyze mode disabled")

    findings = static_findings + llm_findings
    findings = sorted(
        {
            (item.rule_id, item.severity, item.file_path, item.line, item.column, item.message): item
            for item in findings
        }.values(),
        key=lambda item: (item.file_path, item.line, item.column, item.rule_id),
    )
    logger.info(
        "Analyze command completed",
        extra={
            "static_findings": len(static_findings),
            "llm_findings": len(llm_findings),
            "merged_findings": len(findings),
        },
    )

    if as_json:
        _emit_json([finding.to_dict() for finding in findings], output_file=output_file)
    else:
        lines: list[str] = []
        if not findings:
            lines.append("No findings.")
        for finding in findings:
            lines.append(
                f"[{finding.severity}] {finding.rule_id} {finding.file_path}:{finding.line}:{finding.column} - {finding.message}"
            )
        _emit_text("\n".join(lines), output_file=output_file)

    if full_structure:
        logger.debug("Full structure mode enabled", extra={"include_tests": include_tests})
        structure_service = StructureService(include_tests=include_tests)
        try:
            report = structure_service.analyze_project(path)
            logger.debug("Structure analysis completed", extra={"path": path})
        except Exception as exc:
            _log_exception("structure_analyze_project", exc, path=path, include_tests=include_tests)
            raise

        try:
            describe_service = build_describe_service(settings, model_override=model, use_llm=with_llm)
            summary_report = describe_service.describe(
                path,
                max_files=llm_max_files,
                include_functions=False,
                use_llm=with_llm,
            )
            logger.debug("Describe completed", extra={"path": path, "with_llm": with_llm, "max_files": llm_max_files})
        except Exception as exc:
            _log_exception("describe_in_analyze", exc, path=path, with_llm=with_llm)
            raise

        summary_payload = summary_report.to_dict()
        payload = report.to_dict()
        payload["findings"] = [finding.to_dict() for finding in findings]
        payload["summarization"] = summary_payload
        markdown = (
            _render_findings_markdown(findings)
            + "\n\n"
            + structure_service.render_markdown(report).rstrip()
            + "\n\n"
            + _render_repository_summary_markdown(summary_payload)
        )
        if output_format not in {"json", "markdown", "both"}:
            raise typer.BadParameter("--output-format must be one of: json, markdown, both")
        out_json, out_md = _resolve_analyze_artifact_paths(path)
        try:
            if output_format in {"json", "both"}:
                out_json.parent.mkdir(parents=True, exist_ok=True)
                out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                logger.info("Wrote analyze JSON", extra={"out_json": str(out_json)})
            if output_format in {"markdown", "both"}:
                out_md.parent.mkdir(parents=True, exist_ok=True)
                out_md.write_text(markdown, encoding="utf-8")
                logger.info("Wrote analyze Markdown", extra={"out_md": str(out_md)})
        except Exception as exc:
            _log_exception("analyze_write_artifacts", exc, out_json=str(out_json), out_md=str(out_md))
            raise
        if output_format in {"json", "both"}:
            _emit_json(payload, output_file=output_file)
        if output_format in {"markdown", "both"}:
            _emit_text(markdown, output_file=output_file)

    if tech_summary:
        logger.debug("Tech summary mode enabled", extra={"path": path})
        try:
            deps_report = build_dependency_service().analyze(path)
        except Exception as exc:
            _log_exception("deps_analyze_in_tech_summary", exc, path=path)
            raise

        payload = {
            "frameworks": deps_report.frameworks,
            "technologies": deps_report.technologies,
            "package_managers": deps_report.package_managers,
            "languages": deps_report.languages,
            "chain_profile": chain_profile,
            "chain_config": chain_config or "",
        }
        if as_json:
            _emit_json(payload, output_file=output_file)
        else:
            lines = ["Tech Summary:"]
            for key, value in payload.items():
                lines.append(f"- {key}: {value}")
            _emit_text("\n".join(lines), output_file=output_file)

    has_warning = any(item.severity == "warning" for item in findings)
    has_error = any(item.severity == "error" for item in findings)

    logger.debug("Fail-on evaluation", extra={"fail_on": fail_on, "has_warning": has_warning, "has_error": has_error})
    if fail_on == "warning" and (has_warning or has_error):
        raise typer.Exit(code=1)
    if fail_on == "error" and has_error:
        raise typer.Exit(code=1)


@app.command()
def ask(
    question: str,
    k: int | None = typer.Option(None, "--k"),
    model: str | None = typer.Option(None, "--model"),
    index_dir: str | None = typer.Option(None, "--index-dir"),
    ephemeral_index: bool = typer.Option(
        False,
        "--ephemeral-index",
        help="Use temporary index(es) and delete them after answering (ignored if --index-dir is set).",
    ),
    dir_mode: bool = typer.Option(False, "--dir-mode", help="Enable directory-aware ask mode."),
    root_dir: str | None = typer.Option(None, "--root-dir", help="Root directory to scan when --dir-mode is enabled."),
    max_indexes: int = typer.Option(0, "--max-indexes", help="Maximum discovered indexes to use (0 means no limit)."),
    auto_index_missing: bool = typer.Option(
        True,
        "--auto-index-missing/--no-auto-index-missing",
        help="Automatically create missing subproject indexes in --dir-mode.",
    ),
    agent_tools: bool = typer.Option(True, "--agent-tools"),
    agent_max_steps: int = typer.Option(6, "--agent-max-steps"),
    agent_unlimited: bool = typer.Option(
        False,
        "--agent-unlimited/--no-agent-unlimited",
        help="Use effectively unlimited agent tool steps (subject to timeout/resources).",
    ),
    agent_timeout_seconds: int = typer.Option(30, "--agent-timeout-seconds"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    output_file = _resolve_output_file()
    logger.info(
        "Ask command started",
        extra={"question": question, "k": k, "model_override": model, "dir_mode": dir_mode, "index_dir": index_dir, "ephemeral_index": ephemeral_index},
    )
    settings = Settings()
    resolved_k = k or settings.default_top_k
    effective_agent_max_steps = _resolve_agent_max_steps(
        agent_max_steps,
        agent_unlimited=agent_unlimited,
        min_steps=1,
    )

    # For agent tools, project_root matters (file reads)
    # In dir-mode we set to root later; in classic, cwd is fine
    service = _build_ask_service_compat(settings, model_override=model)

    tmp_single: tempfile.TemporaryDirectory | None = None
    tmp_dir_mode_root: tempfile.TemporaryDirectory | None = None

    try:
        if dir_mode:
            root = Path(root_dir).resolve() if root_dir else Path.cwd().resolve()
            if root.is_file():
                root = root.parent

            # rebuild AskService with correct root for tool file reads
            service = _build_ask_service_compat(settings, model_override=model, project_root=root)

            logger.debug(
                "Resolved ask dir-mode parameters",
                extra={
                    "k": resolved_k,
                    "root_dir": str(root),
                    "max_indexes": max_indexes,
                    "auto_index_missing": auto_index_missing,
                    "agent_tools": agent_tools,
                    "agent_unlimited": agent_unlimited,
                    "ephemeral_index": ephemeral_index,
                },
            )

            discovered_subprojects = discover_subprojects(root)
            discovered_indexes = discover_index_dirs(root)
            discovered_index_set = {item.resolve() for item in discovered_indexes}

            index_service = build_index_service(settings)
            auto_indexed_count = 0
            skipped_missing_count = 0
            warnings: list[str] = []
            selected_indexes: list[Path] = []

            tmp_base: Path | None = None
            if ephemeral_index and not index_dir:
                tmp_dir_mode_root, tmp_base = _make_ephemeral_index_dir(prefix="mana_indexes_")

            if discovered_subprojects:
                for subproject in discovered_subprojects:
                    if tmp_base is not None:
                        expected_index = (tmp_base / _stable_subdir_name(subproject.root_path)).resolve()
                        has_index_dir = expected_index.exists()
                    else:
                        expected_index = (subproject.root_path / ".mana_index").resolve()
                        has_index_dir = expected_index in discovered_index_set

                    has_search_data = has_index_dir and _index_has_search_data(expected_index)

                    if has_search_data:
                        selected_indexes.append(expected_index)
                        continue

                    if auto_index_missing:
                        try:
                            logger.info("Auto-indexing missing/empty index", extra={"subproject_root": str(subproject.root_path), "index_dir": str(expected_index)})
                            _index_service_index_compat(
                                index_service,
                                target_path=subproject.root_path,
                                index_dir=expected_index,
                                rebuild=False,
                                vectors=True,
                            )
                            auto_indexed_count += 1
                            selected_indexes.append(expected_index)
                        except Exception as exc:
                            warning = f"Failed to auto-index {subproject.root_path}: {exc}"
                            logger.warning(warning)
                            warnings.append(warning)
                            if _index_has_chunks(expected_index):
                                selected_indexes.append(expected_index)
                    else:
                        skipped_missing_count += 1
                        warning = f"Skipped missing or empty index for subproject {subproject.root_path}"
                        warnings.append(warning)
                        logger.warning(warning)
            else:
                if tmp_base is not None:
                    root_index = (tmp_base / _stable_subdir_name(root)).resolve()
                else:
                    root_index = (root / ".mana_index").resolve()

                if root_index.exists() and _index_has_search_data(root_index):
                    selected_indexes = [root_index]
                elif auto_index_missing:
                    try:
                        logger.info("Auto-indexing root", extra={"root": str(root), "index_dir": str(root_index)})
                        _index_service_index_compat(
                            index_service,
                            target_path=root,
                            index_dir=root_index,
                            rebuild=False,
                            vectors=True,
                        )
                        auto_indexed_count = 1
                        selected_indexes = [root_index]
                    except Exception as exc:
                        warning = f"Failed to auto-index {root}: {exc}"
                        logger.warning(warning)
                        warnings.append(warning)
                        if _index_has_chunks(root_index):
                            selected_indexes = [root_index]

            selected_indexes = sorted({item.resolve() for item in selected_indexes}, key=lambda item: str(item))
            if max_indexes > 0:
                selected_indexes = selected_indexes[:max_indexes]

            logger.info(
                "Ask dir-mode index selection completed",
                extra={
                    "root_dir": str(root),
                    "discovered_indexes": len(discovered_indexes),
                    "selected_indexes": len(selected_indexes),
                    "auto_indexed_count": auto_indexed_count,
                    "skipped_missing_count": skipped_missing_count,
                    "ephemeral_index": ephemeral_index,
                },
            )

            if agent_tools:
                response = service.ask_with_tools_dir_mode(
                    index_dirs=selected_indexes,
                    question=question,
                    k=resolved_k,
                    max_steps=effective_agent_max_steps,
                    timeout_seconds=agent_timeout_seconds,
                    root_dir=root,
                )
            else:
                response = service.ask_dir_mode(index_dirs=selected_indexes, question=question, k=resolved_k, root_dir=root)
            if warnings:
                response.warnings.extend(warnings)

        else:
            if ephemeral_index and not index_dir:
                tmp_single, resolved_index_dir = _make_ephemeral_index_dir()
                index_service = build_index_service(settings)
                _index_service_index_compat(
                    index_service,
                    target_path=Path.cwd(),
                    index_dir=resolved_index_dir,
                    rebuild=False,
                    vectors=True,
                )
            else:
                resolved_index_dir = Path(index_dir).resolve() if index_dir else default_index_dir(Path.cwd())

            logger.debug(
                "Resolved ask parameters",
                extra={"k": resolved_k, "index_dir": str(resolved_index_dir), "agent_tools": agent_tools, "ephemeral_index": ephemeral_index},
            )

            if agent_tools:
                response = service.ask_with_tools(
                    index_dir=resolved_index_dir,
                    question=question,
                    k=resolved_k,
                    max_steps=effective_agent_max_steps,
                    timeout_seconds=agent_timeout_seconds,
                )
            else:
                response = service.ask(index_dir=resolved_index_dir, question=question, k=resolved_k)

        logger.info(
            "Ask command completed",
            extra={"sources": len(response.sources), "mode": getattr(response, "mode", "classic")},
        )

        if as_json:
            _emit_json(response.to_dict(), output_file=output_file)
            return

        lines: list[str] = [response.answer]
        if hasattr(response, "mode"):
            lines.append("")
            lines.append(f"Mode: {response.mode}")
        if hasattr(response, "trace") and response.trace:
            lines.append("")
            lines.append("Tool Trace:")
            for item in response.trace:
                lines.append(
                    f"- {item.tool_name} [{item.status}] {item.duration_ms:.1f}ms args={item.args_summary}"
                )
        if getattr(response, "warnings", None):
            lines.append("")
            lines.append("Warnings:")
            for warning in response.warnings:
                lines.append(f"- {warning}")
        lines.append("")
        lines.append("Sources:")
        if not response.sources:
            lines.append("- none")
            _emit_text("\n".join(lines), output_file=output_file)
            return

        if getattr(response, "source_groups", None):
            for group in response.source_groups:
                lines.append(f"- subproject={group.subproject_root} index={group.index_dir}")
                for source in group.sources:
                    lines.append(f"  - {source.file_path}:{source.start_line}-{source.end_line}")
        else:
            for source in response.sources:
                lines.append(f"- {source.file_path}:{source.start_line}-{source.end_line}")
        _emit_text("\n".join(lines), output_file=output_file)

    finally:
        if tmp_single is not None:
            tmp_single.cleanup()
        if tmp_dir_mode_root is not None:
            tmp_dir_mode_root.cleanup()


@app.command()
def deps(
    path: str,
    model: str | None = typer.Option(None, "--llm-model"),
    use_llm: bool = typer.Option(False, "--llm"),
    rules_file: str | None = typer.Option(None, "--rules"),
    output_json: str | None = typer.Option(None, "--json-out"),
    output_dot: str | None = typer.Option(None, "--dot"),
    output_graphml: str | None = typer.Option(None, "--graphml"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    output_file = _resolve_output_file(path)
    logger.info("Deps command started", extra={"path": path, "use_llm": use_llm, "model_override": model, "rules_file": rules_file})
    settings = Settings()
    dependency_service = build_dependency_service()

    try:
        report = dependency_service.analyze(path)
        logger.debug("Dependency analyze completed", extra={"path": path})
    except Exception as exc:
        _log_exception("deps_analyze", exc, path=path)
        raise

    if use_llm:
        try:
            describe_service = build_describe_service(settings, model_override=model, use_llm=True)
            sample_files = report.manifests[:5]
            samples: list[dict[str, str]] = []
            root = Path(path).resolve()
            if root.is_file():
                root = root.parent
            for rel in sample_files:
                target = root / rel
                if target.exists():
                    samples.append({"file_path": rel, "content": target.read_text(encoding="utf-8", errors="ignore")[:4000]})
            llm_frameworks = describe_service.llm_chain.detect_frameworks_from_samples(samples) if describe_service.llm_chain else []
            report = describe_service.merge_llm_framework_hints(report, llm_frameworks)
            logger.debug("LLM merge completed", extra={"llm_frameworks": llm_frameworks})
        except Exception as exc:
            _log_exception("deps_llm_enrich", exc, path=path, model_override=model)
            raise

    if rules_file:
        try:
            rules = Path(rules_file).read_text(encoding="utf-8", errors="ignore").lower()
            logger.debug("Loaded rules file", extra={"rules_file": rules_file, "len": len(rules)})
            if "django" in rules:
                report.frameworks = sorted(set(report.frameworks) | {"Django"})
                report.technologies = sorted(set(report.technologies) | {"Django"})
            if "react" in rules:
                report.frameworks = sorted(set(report.frameworks) | {"React"})
                report.technologies = sorted(set(report.technologies) | {"React"})
        except Exception as exc:
            _log_exception("deps_rules_file", exc, rules_file=rules_file)
            raise

    try:
        if output_json:
            Path(output_json).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
            logger.info("Wrote deps JSON", extra={"output_json": output_json})
        if output_dot:
            Path(output_dot).write_text(report.to_dot(), encoding="utf-8")
            logger.info("Wrote deps DOT", extra={"output_dot": output_dot})
        if output_graphml:
            Path(output_graphml).write_text(report.to_graphml(), encoding="utf-8")
            logger.info("Wrote deps GraphML", extra={"output_graphml": output_graphml})
    except Exception as exc:
        _log_exception("deps_write_outputs", exc, output_json=output_json, output_dot=output_dot, output_graphml=output_graphml)
        raise

    if as_json:
        _emit_json(report.to_dict(), output_file=output_file)
    else:
        lines = [
            f"Project: {report.project_root}",
            f"Languages: {', '.join(report.languages) if report.languages else 'unknown'}",
            f"Frameworks: {', '.join(report.frameworks) if report.frameworks else 'none'}",
            f"Package managers: {', '.join(report.package_managers) if report.package_managers else 'unknown'}",
            f"Runtime dependencies: {len(report.runtime_dependencies)}",
            f"Dev dependencies: {len(report.dev_dependencies)}",
            f"Module edges: {len(report.module_edges)}",
            f"External edges: {len(report.dependency_edges)}",
        ]
        _emit_text("\n".join(lines), output_file=output_file)


@app.command()
def graph(
    path: str,
    output_dot: str | None = typer.Option(None, "--dot"),
    output_graphml: str | None = typer.Option(None, "--graphml"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    output_file = _resolve_output_file(path)
    logger.info("Graph command started", extra={"path": path, "output_dot": output_dot, "output_graphml": output_graphml})
    try:
        report = build_dependency_service().analyze(path)
        logger.debug("Graph dependency analyze completed", extra={"path": path})
    except Exception as exc:
        _log_exception("graph_deps_analyze", exc, path=path)
        raise

    try:
        if output_dot:
            Path(output_dot).write_text(report.to_dot(), encoding="utf-8")
            logger.info("Wrote graph DOT", extra={"output_dot": output_dot})
        if output_graphml:
            Path(output_graphml).write_text(report.to_graphml(), encoding="utf-8")
            logger.info("Wrote graph GraphML", extra={"output_graphml": output_graphml})
    except Exception as exc:
        _log_exception("graph_write_outputs", exc, output_dot=output_dot, output_graphml=output_graphml)
        raise

    payload = {
        "project_root": report.project_root,
        "module_edges": len(report.module_edges),
        "external_edges": len(report.dependency_edges),
        "dot_output": output_dot or "",
        "graphml_output": output_graphml or "",
    }
    if as_json:
        _emit_json(payload, output_file=output_file)
    else:
        rendered = "\n".join(f"{key}: {value}" for key, value in payload.items())
        _emit_text(rendered, output_file=output_file)


@app.command()
def describe(
    path: str,
    use_llm: bool = typer.Option(True, "--llm/--no-llm"),
    model: str | None = typer.Option(None, "--llm-model"),
    max_files: int = typer.Option(12, "--max-files"),
    functions: bool = typer.Option(False, "--functions"),
    include: str | None = typer.Option(None, "--include", help="Comma-separated include glob patterns."),
    exclude: str | None = typer.Option(None, "--exclude", help="Comma-separated exclude glob patterns."),
    recent_days: int | None = typer.Option(None, "--recent-days", help="Only analyze files modified in last N days."),
    include_docstrings: bool = typer.Option(True, "--docstrings/--no-docstrings"),
    no_cache: bool = typer.Option(False, "--no-cache"),
    output_format: str = typer.Option("both", "--output-format"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    output_file = _resolve_output_file(path)
    logger.info(
        "Describe command started",
        extra={
            "path": path,
            "use_llm": use_llm,
            "model_override": model,
            "max_files": max_files,
            "functions": functions,
            "include": include,
            "exclude": exclude,
            "recent_days": recent_days,
            "include_docstrings": include_docstrings,
            "no_cache": no_cache,
            "output_format": output_format,
        },
    )
    settings = Settings() if use_llm else None
    if use_llm:
        assert settings is not None
        service = build_describe_service(settings, model_override=model, use_llm=True)
    else:
        service = DescribeService(dependency_service=build_dependency_service(), llm_chain=None)

    include_patterns = [item.strip() for item in (include or "").split(",") if item.strip()] or None
    exclude_patterns = [item.strip() for item in (exclude or "").split(",") if item.strip()] or None
    modified_since = None
    if recent_days is not None:
        if recent_days < 0:
            raise typer.BadParameter("--recent-days must be >= 0")
        modified_since = datetime.now() - timedelta(days=recent_days)

    try:
        report = service.describe(
            path,
            max_files=max_files,
            include_functions=functions,
            use_llm=use_llm,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            modified_since=modified_since,
            include_docstrings=include_docstrings,
            use_cache=not no_cache,
        )
        logger.info("Describe completed", extra={"path": path})
    except Exception as exc:
        _log_exception("describe_service.describe", exc, path=path)
        raise

    markdown = service.render_markdown(report)
    if output_format not in {"json", "markdown", "both"}:
        raise typer.BadParameter("--output-format must be one of: json, markdown, both")
    if as_json or output_format in {"json", "both"}:
        _emit_json(report.to_dict(), output_file=output_file)
    if not as_json and output_format in {"markdown", "both"}:
        _emit_text(markdown, output_file=output_file)


@app.command()
def report(
    path: str,
    with_llm: bool = typer.Option(False, "--with-llm"),
    model: str | None = typer.Option(None, "--model"),
    llm_max_files: int = typer.Option(10, "--llm-max-files"),
    summary_max_files: int = typer.Option(12, "--summary-max-files"),
    full_structure: bool = typer.Option(False, "--full-structure"),
    include_tests: bool = typer.Option(False, "--include-tests"),
    online: bool = typer.Option(True, "--online/--offline"),
    osv_timeout_seconds: int = typer.Option(10, "--osv-timeout-seconds"),
    security_scope: str = typer.Option("all", "--security-scope"),
    output_format: str = typer.Option("both", "--output-format"),
    json_out: str | None = typer.Option(None, "--json-out"),
    markdown_out: str | None = typer.Option(None, "--markdown-out"),
    report_profile: str = typer.Option("standard", "--report-profile"),
    detail_line_target: int = typer.Option(350, "--detail-line-target"),
    security_lens: str = typer.Option("defensive-red-team", "--security-lens"),
    as_json: bool = typer.Option(False, "--json", help="Print full JSON report to console."),
) -> None:
    if output_format not in {"json", "markdown", "both"}:
        raise typer.BadParameter("--output-format must be one of: json, markdown, both")
    if security_scope not in {"all", "runtime", "dev"}:
        raise typer.BadParameter("--security-scope must be one of: all, runtime, dev")
    if osv_timeout_seconds <= 0:
        raise typer.BadParameter("--osv-timeout-seconds must be > 0")

    if report_profile not in {"standard", "deep"}:
        raise typer.BadParameter("--report-profile must be standard|deep")
    if security_lens not in {"defensive-red-team", "architecture", "compliance"}:
        raise typer.BadParameter("--security-lens must be defensive-red-team|architecture|compliance")

    if report_profile == "deep":
        detail_line_target = _clamp_detail_line_target(detail_line_target)
        full_structure = True

    report_json_default, report_md_default = _resolve_report_artifact_paths(path)

    out_json = _resolve_out_path(json_out, report_json_default, suffix=".json")
    out_md = _resolve_out_path(markdown_out, report_md_default, suffix=".md")

    logger.info(
        "Report command started",
        extra={
            "path": path,
            "with_llm": with_llm,
            "model_override": model,
            "llm_max_files": llm_max_files,
            "summary_max_files": summary_max_files,
            "full_structure": full_structure,
            "include_tests": include_tests,
            "online": online,
            "osv_timeout_seconds": osv_timeout_seconds,
            "security_scope": security_scope,
            "output_format": output_format,
            "report_profile": report_profile,
            "detail_line_target": detail_line_target,
            "security_lens": security_lens,
            "out_json": str(out_json),
            "out_md": str(out_md),
        },
    )

    service = build_report_service(use_llm=with_llm, model_override=model, include_tests=include_tests)

    try:
        report_obj = service.generate(
            target_path=path,
            with_llm=with_llm,
            model_override=model,
            llm_max_files=llm_max_files,
            summary_max_files=summary_max_files,
            full_structure=full_structure,
            online=online,
            osv_timeout_seconds=osv_timeout_seconds,
            security_scope=security_scope,
            report_profile=report_profile,
            detail_line_target=detail_line_target,
            security_lens=security_lens,
        )
        logger.info("Report service generate completed", extra={"path": path})
    except Exception as exc:
        _log_exception("report_service.generate", exc, path=path)
        raise

    payload = report_obj.to_dict()
    markdown = service.render_markdown(report_obj)

    try:
        if output_format in {"json", "both"}:
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            logger.info("Wrote report JSON", extra={"out_json": str(out_json)})
        if output_format in {"markdown", "both"}:
            out_md.parent.mkdir(parents=True, exist_ok=True)
            out_md.write_text(markdown, encoding="utf-8")
            logger.info("Wrote report Markdown", extra={"out_md": str(out_md)})
    except Exception as exc:
        _log_exception("report_write_artifacts", exc, out_json=str(out_json), out_md=str(out_md))
        raise

    warning_count = len(getattr(report_obj, "warnings", []) or [])
    if as_json:
        console.print_json(json.dumps(payload))
    else:
        lines = [
            "Report generated.",
            f"- JSON: {str(out_json) if output_format in {'json','both'} else '(skipped)'}",
            f"- Markdown: {str(out_md) if output_format in {'markdown','both'} else '(skipped)'}",
            f"- Profile: {report_profile}",
            f"- Warnings: {warning_count}",
        ]
        _emit_text("\n".join(lines), output_file=_resolve_output_file(path))

    logger.info("Report command completed", extra={"warnings": warning_count})


@app.command()
async def scan(
    requirements_file: str = typer.Option(
        "requirements.txt",
        "--requirements-file",
        help="Requirements file used as the baseline for pip/safety scans.",
    ),
    json_out: str | None = typer.Option(
        None,
        "--json-out",
        help="Write a full JSON report of the scan results.",
    ),
    fail_on_vulnerabilities: bool = typer.Option(
        False,
        "--fail-on-vulns",
        help="Return a non-zero exit code if Safety reports vulnerabilities.",
    ),
) -> None:
    output_file = _resolve_output_file()
    req_path = Path(requirements_file).resolve()
    logger.info("Scan command started", extra={"requirements_file": str(req_path), "json_out": json_out, "fail_on_vulns": fail_on_vulnerabilities})

    if not req_path.exists():
        logger.error("requirements file not found", extra={"requirements_file": str(req_path)})
        raise typer.BadParameter(f"requirements file not found: {req_path}")

    console.print(f"[bold]Starting dependency health scan using {req_path.name}[/bold]")
    pip_cmd = [
        sys.executable,
        "-m",
        "pip",
        "list",
        "--outdated",
       	"--format=columns",
        "--disable-pip-version-check",
    ]
    safety_cmd = [
        "safety",
        "check",
        "--full-report",
        "-r",
        str(req_path),
        "--json",
    ]

    logger.debug("Spawning scan subprocesses", extra={"pip_cmd": pip_cmd, "safety_cmd": safety_cmd})
    pip_task = asyncio.create_task(_spawn_command(pip_cmd))
    safety_task = asyncio.create_task(_spawn_command(safety_cmd))

    pip_result, safety_result = await asyncio.gather(pip_task, safety_task, return_exceptions=True)

    def _normalize(result: tuple[int, str, str] | Exception) -> tuple[int, str, str]:
        if isinstance(result, Exception):
            logger.exception("Subprocess task raised", extra={"error": str(result)})
            return -1, "", str(result)
        return result

    pip_code, pip_out, pip_err = _normalize(pip_result)
    safety_code, safety_out, safety_err = _normalize(safety_result)

    report_payload = {
        "requirements_file": str(req_path),
        "pip": {"returncode": pip_code, "stdout": pip_out, "stderr": pip_err},
        "safety": {"returncode": safety_code, "stdout": safety_out, "stderr": safety_err},
    }

    if json_out:
        try:
            Path(json_out).write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
            logger.info("Wrote scan JSON output", extra={"json_out": json_out})
        except Exception as exc:
            _log_exception("scan_write_json", exc, json_out=json_out)
            raise

    console.print("[bold]pip list --outdated[/bold]")
    console.print(_truncate_output(pip_out) or "(no output)")
    if pip_err:
        console.print(f"[red]pip stderr:{_truncate_output(pip_err)}[/red]")

    console.print("[bold]safety check[/bold]")
    console.print(_truncate_output(safety_out) or "(no output)")
    if safety_err:
        console.print(f"[red]safety stderr:{_truncate_output(safety_err)}[/red]")

    summary_lines = [
        f"Dependency scan complete: requirements={req_path}",
        f"- pip exit code: {pip_code}",
        f"- safety exit code: {safety_code}",
    ]
    if safety_code != 0:
        summary_lines.append("- safety reported potential issues (check JSON output)")
    _emit_text("\n".join(summary_lines), output_file=output_file)

    logger.info("Scan command completed", extra={"pip_code": pip_code, "safety_code": safety_code})
    if fail_on_vulnerabilities and safety_code != 0:
        raise typer.Exit(code=1)

# ----------------------------------------------------------------------
# NOTE: index/search/analyze/ask/deps/graph/describe/report/scan omitted
# from this snippet for brevity — keep your existing implementations as-is.
# ----------------------------------------------------------------------

@app.command()
def chat(
    model: str | None = typer.Option(None, "--model"),
    index_dir: str | None = typer.Option(None, "--index-dir"),
    k: int | None = typer.Option(None, "--k"),
    ephemeral_index: bool = typer.Option(
        False,
        "--ephemeral-index",
        help="Use temporary index(es) and delete them when chat exits (ignored if --index-dir is set).",
    ),
    dir_mode: bool = typer.Option(
        False,
        "--dir-mode",
        help="Enable directory-aware chat mode (uses subproject indexes).",
    ),
    root_dir: str | None = typer.Option(
        None,
        "--root-dir",
        help="Root directory to scan when --dir-mode is enabled.",
    ),
    max_indexes: int = typer.Option(
        0,
        "--max-indexes",
        help="Maximum discovered indexes to use in dir-mode (0 means unlimited).",
    ),
    auto_index_missing: bool = typer.Option(
        True,
        "--auto-index-missing/--no-auto-index-missing",
    ),
    tool_worker_process: bool = typer.Option(
        True,
        "--tool-worker-process/--no-tool-worker-process",
        help="Run coding-agent tool execution in a persistent worker subprocess.",
    ),
    tool_worker_strict: bool = typer.Option(
        True,
        "--tool-worker-strict/--no-tool-worker-strict",
        help="Require at least one successful tool call in worker mode.",
    ),
    tool_exec_backend: str = typer.Option(
        "local",
        "--tool-exec-backend",
        help="ToolsManager executor backend: local or redis.",
    ),
    redis_url: str | None = typer.Option(
        None,
        "--redis-url",
        help="Redis URL for --tool-exec-backend redis.",
    ),
    toolsmanager_parallel_requests: int = typer.Option(
        3,
        "--toolsmanager-parallel-requests",
        min=1,
        help="Maximum concurrent per-pass ToolsManager requests.",
    ),
    redis_queue_name: str = typer.Option(
        "mana-tools",
        "--redis-queue-name",
        help="Redis queue name used by ToolsManager redis backend.",
    ),
    redis_ttl_seconds: int = typer.Option(
        86_400,
        "--redis-ttl-seconds",
        min=60,
        help="Redis TTL for ToolsManager runtime status/event keys.",
    ),
    coding_memory: bool = typer.Option(
        True,
        "--coding-memory/--no-coding-memory",
        help="Persist coding-agent flow memory across turns and chat restarts.",
    ),
    flow_id: str | None = typer.Option(
        None,
        "--flow-id",
        help="Optional flow ID to resume or pin coding-agent continuity.",
    ),
    coding_plan_max_steps: int = typer.Option(
        8,
        "--coding-plan-max-steps",
        help="Maximum checklist steps generated by coding planner.",
    ),
    coding_search_budget: int = typer.Option(
        4,
        "--coding-search-budget",
        help="Max semantic_search calls per coding turn.",
    ),
    coding_read_budget: int = typer.Option(
        6,
        "--coding-read-budget",
        help="Max read_file calls per coding turn (full-auto uses this as dynamic per-turn cap).",
    ),
    coding_require_read_files: int = typer.Option(
        2,
        "--coding-require-read-files",
        help="Minimum unique read_file inspections required before edits/answer.",
    ),
    planning_mode: bool = typer.Option(
        False,
        "--planning-mode",
        help="Enable multi-step planning Q&A before generating plan answers.",
    ),
    planning_max_questions: int = typer.Option(
        3,
        "--planning-max-questions",
        help="Maximum planning clarification questions to ask (1-6).",
    ),
    auto_execute_plan: bool = typer.Option(
        False,
        "--auto-execute-plan/--no-auto-execute-plan",
        help="Automatically execute plan-producing turns in agent-tools mode.",
    ),
    auto_execute_max_passes: int = typer.Option(
        4,
        "--auto-execute-max-passes",
        min=1,
        max=12,
        help="Maximum planner->toolsmanager execution passes per turn.",
    ),
    execution_profile: str = typer.Option(
        "balanced",
        "--execution-profile",
        help="Execution profile: full-auto, balanced, conservative.",
    ),
    full_auto: bool = typer.Option(
        False,
        "--full-auto",
        help="Alias for --execution-profile full-auto.",
    ),
    full_auto_status_every: int = typer.Option(
        10,
        "--full-auto-status-every",
        min=0,
        help="In full-auto profile, print a compact checkpoint every N auto-execute passes (0 disables).",
    ),
    agent_max_steps: int = typer.Option(6, "--agent-max-steps"),
    agent_unlimited: bool = typer.Option(
        False,
        "--agent-unlimited/--no-agent-unlimited",
        help="Use effectively unlimited agent tool steps (subject to timeout/resources).",
    ),
    agent_timeout_seconds: int = typer.Option(30, "--agent-timeout-seconds"),
    multiline_input: bool = typer.Option(
        True,
        "--multiline-input/--no-multiline-input",
        help="Enable multiline chat input (`/paste` trigger or buffered paste burst detection).",
    ),
    multiline_terminator: str = typer.Option(
        ".end",
        "--multiline-terminator",
        help="Terminator line used to submit multiline input.",
    ),
    diagram_render_images: bool = typer.Option(
        True,
        "--diagram-render-images/--no-diagram-render-images",
        help="Render Mermaid diagram blocks to SVG/PNG artifacts when possible.",
    ),
    diagram_output_dir: str | None = typer.Option(
        None,
        "--diagram-output-dir",
        help="Directory for rendered diagram artifacts (default: <root>/.mana_diagrams).",
    ),
    diagram_format: str = typer.Option(
        "svg",
        "--diagram-format",
        help="Diagram artifact format: svg or png.",
    ),
    diagram_open: bool = typer.Option(
        False,
        "--diagram-open/--no-diagram-open",
        help="Open rendered diagram artifacts with the system default app.",
    ),
    diagram_timeout_seconds: int = typer.Option(
        25,
        "--diagram-timeout-seconds",
        help="Timeout in seconds for Mermaid artifact rendering.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit responses as JSON objects."),
) -> None:
    output_file = _resolve_output_file()
    agent_tools = True
    coding_agent = True
    tool_worker_process = True
    logger.info(
        "Chat command started",
        extra={
            "model_override": model,
            "index_dir": index_dir,
            "k": k,
            "dir_mode": dir_mode,
            "root_dir": root_dir,
            "max_indexes": max_indexes,
            "auto_index_missing": auto_index_missing,
            "agent_tools": agent_tools,
            "coding_agent": coding_agent,
            "tool_worker_process": tool_worker_process,
            "tool_worker_strict": tool_worker_strict,
            "tool_exec_backend": tool_exec_backend,
            "redis_url": redis_url,
            "toolsmanager_parallel_requests": toolsmanager_parallel_requests,
            "redis_queue_name": redis_queue_name,
            "redis_ttl_seconds": redis_ttl_seconds,
            "coding_memory": coding_memory,
            "flow_id": flow_id,
            "coding_plan_max_steps": coding_plan_max_steps,
            "coding_search_budget": coding_search_budget,
            "coding_read_budget": coding_read_budget,
            "coding_require_read_files": coding_require_read_files,
            "planning_mode": planning_mode,
            "planning_max_questions": planning_max_questions,
            "auto_execute_plan": auto_execute_plan,
            "auto_execute_max_passes": auto_execute_max_passes,
            "execution_profile": execution_profile,
            "full_auto": full_auto,
            "full_auto_status_every": full_auto_status_every,
            "agent_max_steps": agent_max_steps,
            "agent_unlimited": agent_unlimited,
            "agent_timeout_seconds": agent_timeout_seconds,
            "multiline_input": multiline_input,
            "multiline_terminator": multiline_terminator,
            "diagram_render_images": diagram_render_images,
            "diagram_output_dir": diagram_output_dir,
            "diagram_format": diagram_format,
            "diagram_open": diagram_open,
            "diagram_timeout_seconds": diagram_timeout_seconds,
            "as_json": as_json,
            "ephemeral_index": ephemeral_index,
        },
    )
    as_json = False
    if full_auto:
        execution_profile = "full-auto"
    execution_profile = str(execution_profile or "balanced").strip().lower()
    if execution_profile not in {"full-auto", "balanced", "conservative"}:
        raise typer.BadParameter("--execution-profile must be one of: full-auto, balanced, conservative.")

    if execution_profile == "full-auto":
        auto_execute_plan = True
        # Keep user override when they pass a non-default value.
        if int(auto_execute_max_passes) == 4:
            auto_execute_max_passes = 10
    full_auto_status_every = max(0, int(full_auto_status_every))
    planning_question_limit = max(1, min(planning_max_questions, 6))
    auto_execute_max_passes = max(1, min(int(auto_execute_max_passes), 12))
    chat_agent_max_steps = _resolve_agent_max_steps(
        agent_max_steps,
        agent_unlimited=agent_unlimited,
        min_steps=1,
    )
    coding_agent_max_steps = _resolve_agent_max_steps(
        agent_max_steps,
        agent_unlimited=agent_unlimited,
        min_steps=8,
        cap=200,
    )
    # CodingAgent is the sole decision-maker in chat – always force on.

    if auto_execute_plan:
        pass  # already forced above
    settings = Settings()
    resolved_tool_exec_backend = str(
        (tool_exec_backend or getattr(settings, "tool_exec_backend", "local")) or "local"
    ).strip().lower()
    if resolved_tool_exec_backend not in {"local", "redis"}:
        raise typer.BadParameter("--tool-exec-backend must be 'local' or 'redis'.")
    resolved_redis_url = str(
        (redis_url or getattr(settings, "redis_url", "redis://127.0.0.1:6379/0"))
        or "redis://127.0.0.1:6379/0"
    ).strip()
    resolved_parallel_requests = max(
        1,
        int(
            toolsmanager_parallel_requests
            or getattr(settings, "toolsmanager_parallel_requests", 3)
            or 3
        ),
    )
    resolved_redis_queue_name = str(
        (redis_queue_name or getattr(settings, "redis_queue_name", "mana-tools")) or "mana-tools"
    ).strip() or "mana-tools"
    resolved_redis_ttl_seconds = max(
        60,
        int(redis_ttl_seconds or getattr(settings, "redis_ttl_seconds", 86_400) or 86_400),
    )
    tools_execution_config = ToolsExecutionConfig(
        backend=resolved_tool_exec_backend,
        redis_url=resolved_redis_url,
        queue_name=resolved_redis_queue_name,
        parallel_requests=resolved_parallel_requests,
        ttl_seconds=resolved_redis_ttl_seconds,
    )
    tools_execution_boot_warnings: list[str] = []
    diagram_format = str(diagram_format or "svg").strip().lower()
    if diagram_format not in {"svg", "png"}:
        raise typer.BadParameter("--diagram-format must be 'svg' or 'png'.")
    diagram_timeout_seconds = max(5, int(diagram_timeout_seconds))
    multiline_terminator = str(multiline_terminator or ".end").strip()
    if not multiline_terminator:
        raise typer.BadParameter("--multiline-terminator must be a non-empty line token.")
    resolved_k = k or settings.default_top_k

    if dir_mode:
        root = Path(root_dir).resolve() if root_dir else Path.cwd().resolve()
    else:
        root = Path.cwd().resolve()
    if root.is_file():
        root = root.parent

    logger.debug("Resolved chat root", extra={"root": str(root)})
    run_logger = LlmRunLogger(
        log_file=(
            root
            / ".mana_llm_logs"
            / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}-{(root.name or 'project')}-runs.jsonl"
        )
    )
    resolved_diagram_output_dir = (
        Path(diagram_output_dir).expanduser().resolve()
        if diagram_output_dir
        else (root / ".mana_diagrams").resolve()
    )

    ask_service = _build_ask_service_compat(settings, model_override=model, project_root=root)

    chat_service = ChatService(
        ask_service=ask_service,
        settings=settings,
        model_override=model,
        index_dir=index_dir,
        dir_mode=dir_mode,
        root_dir=str(root),
        k=resolved_k,
        agent_tools=agent_tools,
        agent_max_steps=chat_agent_max_steps,
        agent_timeout_seconds=agent_timeout_seconds,
        max_indexes=max_indexes,
        auto_index_missing=auto_index_missing,
    )

    # ✅ Initialize CodingAgent (only when enabled)
    coding_agent_instance: CodingAgent | None = None
    coding_memory_service: CodingMemoryService | None = None
    tool_worker_client: ToolWorkerClient | None = None
    tools_manager_orchestrator: ToolsManagerOrchestrator | None = None
    tools_executor_instance: LocalToolsExecutor | RedisRQToolsExecutor | None = None
    effective_model = model or settings.openai_chat_model

    def _build_tools_executor(worker_client: ToolWorkerClient):
        if tools_execution_config.backend != "redis":
            return LocalToolsExecutor(worker_client=worker_client)
        try:
            return RedisRQToolsExecutor(
                worker_init_payload=worker_client.init_payload_dict(),
                config=tools_execution_config,
            )
        except Exception as exc:
            warning = f"redis executor unavailable; falling back to local backend: {exc}"
            logger.warning(warning)
            tools_execution_boot_warnings.append(warning)
            return LocalToolsExecutor(worker_client=worker_client)

    if coding_agent:
        if not agent_tools:
            raise typer.BadParameter("--coding-agent requires --agent-tools (needs tool loop).")
        if getattr(ask_service, "ask_agent", None) is None:
            raise typer.BadParameter("--coding-agent requires AskService.ask_agent to be configured.")
        if coding_memory:
            coding_memory_service = CodingMemoryService(
                project_root=root,
                max_turns=settings.coding_flow_max_turns,
                max_tasks=settings.coding_flow_max_tasks,
            )
        if tool_worker_process:
            tool_worker_client = ToolWorkerClient(
                api_key=settings.openai_api_key,
                model=effective_model,
                base_url=settings.openai_base_url,
                embed_model=settings.openai_embed_model,
                repo_root=root,
                project_root=root,
                allowed_prefixes=None,
                tools_only_strict=tool_worker_strict,
            )

        # ✅ IMPORTANT: allow_prefixes=None => unrestricted under repo_root
        coding_agent_instance = CodingAgent(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            repo_root=root,
            ask_agent=ask_service.ask_agent,
            allowed_prefixes=None,
            coding_memory_service=coding_memory_service,
            coding_memory_enabled=coding_memory,
            plan_max_steps=max(1, int(coding_plan_max_steps or settings.coding_plan_max_steps)),
            search_budget=max(1, int(coding_search_budget or settings.coding_search_budget)),
            read_budget=max(1, int(coding_read_budget or settings.coding_read_budget)),
            require_read_files=max(1, int(coding_require_read_files or settings.coding_require_read_files)),
            repo_only_internet_default=True,
            tool_worker_client=tool_worker_client,
            full_auto_mode=(execution_profile == "full-auto"),
        )

    if coding_agent_instance is not None and auto_execute_plan and tool_worker_client is not None:
        tools_executor_instance = _build_tools_executor(tool_worker_client)
        tools_manager_orchestrator = ToolsManagerOrchestrator(
            api_key=settings.openai_api_key,
            model=effective_model,
            base_url=settings.openai_base_url,
            worker_client=tool_worker_client,
            repo_root=root,
            execution_config=tools_execution_config,
            executor=tools_executor_instance,
        )
        if hasattr(coding_agent_instance, "set_tools_manager_orchestrator"):
            coding_agent_instance.set_tools_manager_orchestrator(tools_manager_orchestrator)

    tmp_root: tempfile.TemporaryDirectory | None = None
    tmp_base: Path | None = None
    dir_mode_index_dirs: list[Path] = []
    try:
        # -----------------------------
        # Resolve indexes (dir-mode vs classic)
        # -----------------------------
        resolved_index_dir: Path | None = None

        if dir_mode:
            discovered_subprojects = discover_subprojects(root)
            discovered_indexes = discover_index_dirs(root)
            discovered_index_set = {item.resolve() for item in discovered_indexes}

            if ephemeral_index and not index_dir:
                tmp_root, tmp_base = _make_ephemeral_index_dir(prefix="mana_chat_indexes_")

            index_service = build_index_service(settings)
            auto_indexed_count = 0
            skipped_missing_count = 0
            warnings: list[str] = []
            selected_indexes: list[Path] = []

            def _auto_index(target_path: Path, idx_dir: Path) -> bool:
                logger.info("Chat auto-index attempt", extra={"target_path": str(target_path), "idx_dir": str(idx_dir)})
                try:
                    _index_service_index_compat(
                        index_service,
                        target_path=target_path,
                        index_dir=idx_dir,
                        rebuild=False,
                        vectors=True,
                    )
                    logger.info("Chat auto-index vectors success", extra={"idx_dir": str(idx_dir)})
                    return True
                except Exception as exc:
                    warning = f"Vector index failed for {target_path} (fallback to chunks-only): {exc}"
                    logger.warning(warning)
                    warnings.append(warning)
                    try:
                        _index_service_index_compat(
                            index_service,
                            target_path=target_path,
                            index_dir=idx_dir,
                            rebuild=False,
                            vectors=False,
                        )
                        logger.info("Chat auto-index chunks-only success", extra={"idx_dir": str(idx_dir)})
                        return True
                    except Exception as exc2:
                        warning2 = f"Chunks-only index failed for {target_path}: {exc2}"
                        logger.warning(warning2)
                        warnings.append(warning2)
                        return False

            if discovered_subprojects:
                for subproject in discovered_subprojects:
                    if tmp_base is not None:
                        expected_index = (tmp_base / _stable_subdir_name(subproject.root_path)).resolve()
                        has_index_dir = expected_index.exists()
                    else:
                        expected_index = (subproject.root_path / ".mana_index").resolve()
                        has_index_dir = expected_index in discovered_index_set

                    has_search_data = has_index_dir and _index_has_search_data(expected_index)

                    if has_search_data:
                        selected_indexes.append(expected_index)
                        continue

                    if auto_index_missing:
                        ok = _auto_index(subproject.root_path, expected_index)
                        if ok:
                            auto_indexed_count += 1
                            selected_indexes.append(expected_index)
                        else:
                            if _index_has_chunks(expected_index):
                                selected_indexes.append(expected_index)
                    else:
                        skipped_missing_count += 1
                        warning = f"Skipped missing or empty index for subproject {subproject.root_path}"
                        logger.warning(warning)
                        warnings.append(warning)

            else:
                if tmp_base is not None:
                    root_index = (tmp_base / _stable_subdir_name(root)).resolve()
                else:
                    root_index = (root / ".mana_index").resolve()

                if root_index.exists() and _index_has_search_data(root_index):
                    selected_indexes = [root_index]
                elif auto_index_missing:
                    ok = _auto_index(root, root_index)
                    if ok:
                        auto_indexed_count = 1
                        selected_indexes = [root_index]
                    else:
                        if _index_has_chunks(root_index):
                            selected_indexes = [root_index]

            selected_indexes = sorted({item.resolve() for item in selected_indexes}, key=lambda p: str(p))
            if max_indexes > 0:
                selected_indexes = selected_indexes[:max_indexes]

            if not selected_indexes:
                msg = (
                    f"No usable indexes found under {root}. "
                    f"Try running: mana-analyzer index {root} "
                    f"or re-run chat with --auto-index-missing."
                )
                _emit_text(msg, output_file=output_file)
                logger.error("Chat aborted: no usable indexes", extra={"root": str(root)})
                raise typer.Exit(code=2)

            chat_service.set_index_dirs(selected_indexes)
            dir_mode_index_dirs = selected_indexes
            
            _emit_text(
                "mana-analyzer chat (dir-mode)\n"
                f"- root: {root}\n"
                f"- indexes: {len(selected_indexes)}\n"
                f"- auto-indexed: {auto_indexed_count}\n"
                f"- skipped: {skipped_missing_count}\n"
                f"- ephemeral-index: {ephemeral_index}\n"
                f"- coding-agent: {coding_agent}\n"
                f"- coding-memory: {coding_memory and coding_agent}",
                output_file=output_file,
            )
            if warnings:
                _emit_text("Warnings:\n" + "\n".join(f"- {w}" for w in warnings), output_file=output_file)

        else:
            if ephemeral_index and not index_dir:
                tmp_root, resolved_index_dir = _make_ephemeral_index_dir(prefix="mana_chat_index_")
                index_service = build_index_service(settings)
                _index_service_index_compat(
                    index_service,
                    target_path=Path.cwd(),
                    index_dir=resolved_index_dir,
                    rebuild=False,
                    vectors=True,
                )
            else:
                resolved_index_dir = Path(index_dir).resolve() if index_dir else default_index_dir(Path.cwd())

            _emit_text(
                "mana-analyzer chat\n"
                f"- index: {resolved_index_dir}\n"
                f"- k: {resolved_k}\n"
                f"- ephemeral-index: {ephemeral_index}\n"
                f"- coding-agent: {coding_agent}\n"
                f"- coding-memory: {coding_memory and coding_agent}",
                output_file=output_file,
            )

        # -----------------------------
        # Tool-first retry logic
        # -----------------------------
        def _looks_like_guess(answer: str) -> bool:
            a = (answer or "").lower()
            triggers = [
                "i can't",
                "cannot",
                "can't",
                "no source",
                "no actual source",
                "snippets you provided",
                "from the snippets",
                "i don’t have",
                "i don't have",
                "not enough",
                "insufficient",
                "based on the repository name",
                "infer",
                "guess",
                "might be",
                "appears to",
                "only clues",
                "doesn’t contain application logic",
                "does not contain application logic",
                "path is outside project root",
            ]
            return any(t in a for t in triggers)

        def _tool_first_instruction(user_question: str) -> str:
            return (
                f"{user_question}\n\n"
                "TOOL-FIRST INSTRUCTIONS (MANDATORY):\n"
                "- Do NOT guess.\n"
                "- You MUST use tools to search the repository and inspect real source files.\n"
                "- You MUST open at least TWO real source files before concluding.\n"
                "- Avoid caches/build output: node_modules/, .next/, .angular/, dist/, build/, .cache/, .npm-cache/, generated/.\n"
                "- Final answer MUST include evidence (file path + line ranges).\n"
            )

        max_attempts = 3
        min_sources = 2

        console.print("mana-analyzer chat – type 'exit' or 'quit' to end.")

        if planning_mode:
            console.print(
                f"[bold cyan]Planning mode enabled:[/bold cyan] "
                f"up to {planning_question_limit} clarification question(s) before each plan answer."
            )

        # CodingAgent is always active
        console.print("[bold red]Coding agent active[/bold red] – all decisions routed through CodingAgent.")
        _ca_flow = flow_id or (coding_agent_instance.get_active_flow_id() if coding_agent_instance else None)
        if coding_memory:
            if _ca_flow:
                console.print(f"[cyan]Flow memory:[/cyan] resuming flow {_ca_flow}")
            else:
                console.print("[cyan]Flow memory:[/cyan] new flow will be created on first request.")
        if True:  # agent_tools always on
            console.print(
                f"[cyan]Auto-execute:[/cyan] {'enabled' if auto_execute_plan else 'disabled'} "
                f"(max passes: {auto_execute_max_passes})"
            )
            console.print(f"[cyan]Execution profile:[/cyan] {execution_profile}")
            if execution_profile == "full-auto":
                if full_auto_status_every > 0:
                    console.print(
                        f"[cyan]Full-auto checkpoint:[/cyan] every {full_auto_status_every} pass(es)"
                    )
                else:
                    console.print("[cyan]Full-auto checkpoint:[/cyan] disabled")
        if diagram_render_images:
            console.print(
                f"[cyan]Diagram rendering:[/cyan] {diagram_format} -> {resolved_diagram_output_dir}"
            )
        if tool_worker_client is not None and (coding_agent_instance is not None or tools_manager_orchestrator is not None):
            try:
                tool_worker_client.start()
                tool_worker_client.health()
                console.print(
                    "[cyan]Tool worker subprocess active:[/cyan] "
                    f"tools-only strict={tool_worker_strict}; "
                    f"exec-backend={tools_execution_config.backend}; "
                    f"parallel={tools_execution_config.parallel_requests}"
                )
            except Exception as exc:
                _log_exception("tool_worker_client.start", exc)
                console.print(
                    "[yellow]Tool worker failed to initialize. Auto-execute and coding-agent paths are disabled for this session.[/yellow]"
                )
                coding_agent_instance = None
                tools_manager_orchestrator = None

        planning_request: str | None = None
        planning_answers: list[str] = []
        planning_questions: list[str] = []
        planning_question_source: str = "none"
        planning_questions_asked_count: int = 0
        active_flow_id: str | None = flow_id
        pending_conflict_question: str | None = None
        pending_ui_selection: dict[str, Any] | None = None
        pending_prechecklist: dict[str, Any] | None = None
        pending_prechecklist_source: str = ""
        pending_prechecklist_warning: str = ""
        session_turns: list[ChatTurnTelemetry] = []
        full_auto_pass_window_logs: list[dict[str, Any]] = []
        full_auto_pass_window_decisions: list[dict[str, Any]] = []
        full_auto_latest_checklist_counts: dict[str, int] | None = None
        full_auto_passes_total: int = 0
        full_auto_pass_checkpoints_emitted: int = 0

        def _base_auto_execute_tool_policy(user_question: str) -> dict[str, Any]:
            if coding_agent_instance is not None:
                return coding_agent_instance._tool_policy_for_request(user_question)  # type: ignore[attr-defined]
            block_internet = not bool(re.search(r"(?i)\\b(latest|internet|online|web|news|search web)\\b", user_question))
            return {
                "allowed_tools": [
                    "semantic_search",
                    "read_file",
                    "run_command",
                    "apply_patch",
                    "write_file",
                    "search_internet",
                ],
                "search_budget": max(1, int(coding_search_budget or settings.coding_search_budget)),
                "read_budget": max(1, int(coding_read_budget or settings.coding_read_budget)),
                "read_line_window": 400,
                "require_read_files": max(1, int(coding_require_read_files or settings.coding_require_read_files)),
                "block_internet": block_internet,
                "search_repeat_limit": 1,
                "max_semantic_k": 50,
            }

        def _ensure_tools_manager_orchestrator() -> ToolsManagerOrchestrator | None:
            nonlocal tool_worker_client
            nonlocal tools_manager_orchestrator
            nonlocal tools_executor_instance
            if not (agent_tools and auto_execute_plan and tool_worker_process):
                return None
            if tools_manager_orchestrator is not None:
                return tools_manager_orchestrator
            if tool_worker_client is None:
                tool_worker_client = ToolWorkerClient(
                    api_key=settings.openai_api_key,
                    model=model or settings.openai_chat_model,
                    base_url=settings.openai_base_url,
                    embed_model=settings.openai_embed_model,
                    repo_root=root,
                    project_root=root,
                    allowed_prefixes=None,
                    tools_only_strict=tool_worker_strict,
                )
            try:
                tool_worker_client.start()
                tool_worker_client.health()
            except Exception as exc:
                _log_exception("tool_worker_client.start", exc)
                return None
            if tools_executor_instance is None:
                tools_executor_instance = _build_tools_executor(tool_worker_client)
            tools_manager_orchestrator = ToolsManagerOrchestrator(
                api_key=settings.openai_api_key,
                model=model or settings.openai_chat_model,
                base_url=settings.openai_base_url,
                worker_client=tool_worker_client,
                repo_root=root,
                execution_config=tools_execution_config,
                executor=tools_executor_instance,
                coding_memory_service=(
                    getattr(coding_agent_instance, "coding_memory_service", None)
                    if coding_agent_instance is not None
                    else None
                ),
            )
            if coding_agent_instance is not None and hasattr(coding_agent_instance, "set_tools_manager_orchestrator"):
                coding_agent_instance.set_tools_manager_orchestrator(tools_manager_orchestrator)
            return tools_manager_orchestrator

        def _run_auto_execute_pipeline(
            user_question: str,
            *,
            render_progress: bool = True,
        ) -> tuple[dict[str, Any], str]:
            if not agent_tools or not auto_execute_plan:
                return {}, ""
            orchestrator = _ensure_tools_manager_orchestrator()
            if orchestrator is None:
                return {
                    "answer": "Auto-execute requested but tools manager worker is unavailable.",
                    "warnings": [
                        "auto_execute_worker_unavailable",
                        *[str(item).strip() for item in tools_execution_boot_warnings if str(item).strip()],
                    ],
                    "trace": [],
                    "sources": [],
                    "changed_files": [],
                    "plan": None,
                    "passes": 0,
                    "terminal_reason": "worker_unavailable",
                    "toolsmanager_requests_count": 0,
                    "pass_logs": [],
                    "planner_decisions": [],
                    "prechecklist": None,
                    "prechecklist_source": "",
                    "prechecklist_warning": "",
                }, ""
            if dir_mode:
                if not dir_mode_index_dirs:
                    return {
                        "answer": "Auto-execute unavailable: no dir-mode indexes resolved.",
                        "warnings": [
                            "auto_execute_missing_index_dirs",
                            *[str(item).strip() for item in tools_execution_boot_warnings if str(item).strip()],
                        ],
                        "trace": [],
                        "sources": [],
                        "changed_files": [],
                        "plan": None,
                        "passes": 0,
                        "terminal_reason": "missing_indexes",
                        "toolsmanager_requests_count": 0,
                        "pass_logs": [],
                        "planner_decisions": [],
                        "prechecklist": None,
                        "prechecklist_source": "",
                        "prechecklist_warning": "",
                    }, ""
                target_index_dir: Path | None = None
                target_index_dirs: list[Path] | None = list(dir_mode_index_dirs)
            else:
                target_index_dir = resolved_index_dir
                target_index_dirs = None

            flow_context_text: str | None = None
            if coding_agent_instance is not None and active_flow_id:
                try:
                    summary = coding_agent_instance.flow_summary(active_flow_id)
                except Exception:
                    summary = None
                if isinstance(summary, dict):
                    lines: list[str] = []
                    objective = str(summary.get("objective", "") or "").strip()
                    if objective:
                        lines.append(f"Current objective: {objective}")
                    checklist = summary.get("checklist")
                    if isinstance(checklist, dict):
                        steps = checklist.get("steps") if isinstance(checklist.get("steps"), list) else []
                        if steps:
                            lines.append("Current checklist:")
                            for step in steps[:20]:
                                if not isinstance(step, dict):
                                    continue
                                status = str(step.get("status", "pending") or "pending")
                                title = str(step.get("title", "step") or "step")
                                lines.append(f"- [{status}] {title}")
                    if lines:
                        flow_context_text = "\n".join(lines)

            preview_payload: dict[str, Any] = {}
            if hasattr(orchestrator, "preview_plan"):
                try:
                    preview_payload = orchestrator.preview_plan(
                        request=user_question,
                        flow_context=flow_context_text,
                        pass_cap=auto_execute_max_passes,
                    )
                except Exception as exc:
                    _log_exception("tools_manager.preview_plan", exc)
                    preview_payload = {
                        "prechecklist": None,
                        "prechecklist_source": "",
                        "prechecklist_warning": f"Planner preview failed: {exc}",
                        "warnings": [],
                    }
            else:
                preview_payload = {
                    "prechecklist": None,
                    "prechecklist_source": "",
                    "prechecklist_warning": "",
                    "warnings": [],
                }
            preview_checklist = (
                preview_payload.get("prechecklist")
                if isinstance(preview_payload.get("prechecklist"), dict)
                else None
            )
            preview_warning = str(preview_payload.get("prechecklist_warning", "") or "").strip()
            if render_progress and preview_checklist is not None:
                _render_prechecklist_preview(
                    console,
                    prechecklist=preview_checklist,
                    warning=preview_warning,
                )

            if render_progress:
                console.print(
                    f"[cyan]Auto-executing plan:[/cyan] max passes {auto_execute_max_passes} (same turn, no extra confirmation)."
                )

            def _call(callbacks: list[BaseCallbackHandler]):
                _ = callbacks
                return orchestrator.run(
                    request=user_question,
                    flow_context=flow_context_text,
                    index_dir=target_index_dir,
                    index_dirs=target_index_dirs,
                    k=resolved_k,
                    max_steps=chat_agent_max_steps,
                    timeout_seconds=agent_timeout_seconds,
                    tool_policy=_base_auto_execute_tool_policy(user_question),
                    pass_cap=auto_execute_max_passes,
                    on_event=CodingAgent._log_worker_event,
                    flow_id=active_flow_id,
                )

            result_obj, debug_tail = _run_with_live_buffer(
                console,
                spinner_text="Auto-executing…",
                fn=_call,
                callbacks=[],
            )
            if hasattr(result_obj, "model_dump"):
                payload = result_obj.model_dump()
            elif isinstance(result_obj, dict):
                payload = dict(result_obj)
            else:
                payload = {"answer": str(result_obj)}
            merged_executor_warnings = [str(item).strip() for item in tools_execution_boot_warnings if str(item).strip()]
            existing_payload_warnings = (
                [str(item).strip() for item in payload.get("warnings", []) if str(item).strip()]
                if isinstance(payload.get("warnings"), list)
                else []
            )
            payload["warnings"] = [*existing_payload_warnings, *merged_executor_warnings]
            payload["prechecklist"] = preview_checklist
            payload["prechecklist_source"] = str(preview_payload.get("prechecklist_source", "") or "")
            payload["prechecklist_warning"] = preview_warning
            plan_payload = payload.get("plan") if isinstance(payload.get("plan"), dict) else {}
            objective = str(plan_payload.get("objective", "")).strip()
            if render_progress:
                for item in payload.get("pass_logs", []) if isinstance(payload.get("pass_logs"), list) else []:
                    if not isinstance(item, dict):
                        continue
                    _render_auto_execute_pass_status(
                        console,
                        objective=objective,
                        pass_index=int(item.get("pass_index", 0) or 0),
                        pass_cap=auto_execute_max_passes,
                        planner_step_id=str(item.get("planner_step_id", "") or ""),
                        planner_step_title=str(item.get("planner_step_title", "") or ""),
                        planner_decision=str(item.get("planner_decision", "") or ""),
                        planner_decision_reason=str(item.get("planner_decision_reason", "") or ""),
                        batch_reason=str(item.get("batch_reason", "") or ""),
                        expected_progress=str(item.get("expected_progress", "") or ""),
                    )
            return payload, debug_tail

        def _build_full_auto_resume_request(
            *,
            original_question: str,
            resume_cycle: int,
            terminal_reason: str,
        ) -> str:
            reason = str(terminal_reason or "").strip() or "pass_cap_reached"
            return (
                f"{original_question}\n\n"
                "FULL-AUTO RESUME DIRECTIVE:\n"
                f"- resume_cycle: {int(resume_cycle)}\n"
                f"- prior_terminal_reason: {reason}\n"
                "- Continue from current repository/flow state.\n"
                "- Do not ask for confirmation or scope-choice questions.\n"
                "- Execute inspect/edit/verify loops until complete or hard-blocked.\n"
            )

        def _ingest_full_auto_pass_payload(payload: dict[str, Any]) -> int:
            nonlocal full_auto_pass_window_logs
            nonlocal full_auto_pass_window_decisions
            nonlocal full_auto_latest_checklist_counts
            nonlocal full_auto_passes_total

            pass_logs = payload.get("pass_logs") if isinstance(payload.get("pass_logs"), list) else []
            normalized_logs = [item for item in pass_logs if isinstance(item, dict)]
            if normalized_logs:
                full_auto_pass_window_logs.extend(normalized_logs)
                full_auto_passes_total += len(normalized_logs)

            planner_rows = (
                payload.get("planner_decisions")
                if isinstance(payload.get("planner_decisions"), list)
                else []
            )
            normalized_planner_rows = [item for item in planner_rows if isinstance(item, dict)]
            if normalized_planner_rows:
                full_auto_pass_window_decisions.extend(normalized_planner_rows)

            counts = _resolve_payload_checklist_counts(payload)
            if counts is not None:
                full_auto_latest_checklist_counts = counts
            return len(normalized_logs)

        def _emit_full_auto_pass_checkpoints(*, resume_cycles: int) -> int:
            nonlocal full_auto_pass_window_logs
            nonlocal full_auto_pass_window_decisions
            nonlocal full_auto_pass_checkpoints_emitted

            if str(execution_profile or "").strip().lower() != "full-auto":
                return 0
            every = max(0, int(full_auto_status_every))
            if every <= 0:
                return 0

            emitted = 0
            while len(full_auto_pass_window_logs) >= every:
                checkpoint_logs = full_auto_pass_window_logs[:every]
                full_auto_pass_window_logs = full_auto_pass_window_logs[every:]

                take_rows = min(len(full_auto_pass_window_decisions), every)
                checkpoint_planner_rows = full_auto_pass_window_decisions[:take_rows]
                full_auto_pass_window_decisions = full_auto_pass_window_decisions[take_rows:]

                decision_rows = _checkpoint_decisions_from_pass_window(
                    checkpoint_planner_rows,
                    checkpoint_logs,
                )
                _render_full_auto_checkpoint(
                    console,
                    decision_rows=decision_rows,
                    checklist_counts=full_auto_latest_checklist_counts,
                    window_passes=len(checkpoint_logs),
                    pass_total=full_auto_passes_total,
                    resume_cycles=resume_cycles,
                )
                full_auto_pass_checkpoints_emitted += 1
                emitted += 1
            return emitted

        def _run_auto_execute_pipeline_with_resume(
            user_question: str,
            *,
            render_progress: bool = True,
        ) -> tuple[dict[str, Any], str]:
            run_full_auto = str(execution_profile or "").strip().lower() == "full-auto"
            resume_cycles = 0
            resumed_from_pass_cap = False
            turn_passes_total = 0
            turn_checkpoints_emitted = 0
            effective_question = user_question
            last_debug_tail = ""

            while True:
                payload, debug_tail = _run_auto_execute_pipeline(
                    effective_question,
                    render_progress=(render_progress and resume_cycles == 0),
                )
                last_debug_tail = debug_tail
                if run_full_auto:
                    turn_passes_total += _ingest_full_auto_pass_payload(payload)
                terminal_reason = str(payload.get("terminal_reason", "") or "").strip().lower()
                if run_full_auto and terminal_reason == "pass_cap_reached":
                    resumed_from_pass_cap = True
                    resume_cycles += 1
                    turn_checkpoints_emitted += _emit_full_auto_pass_checkpoints(resume_cycles=resume_cycles)
                    effective_question = _build_full_auto_resume_request(
                        original_question=user_question,
                        resume_cycle=resume_cycles,
                        terminal_reason=terminal_reason,
                    )
                    continue

                payload["full_auto_resume_cycles"] = int(resume_cycles)
                payload["full_auto_passes_total"] = int(turn_passes_total)
                payload["full_auto_pass_checkpoints_emitted"] = int(turn_checkpoints_emitted)
                payload["resumed_from_pass_cap"] = bool(resumed_from_pass_cap)
                return payload, last_debug_tail

        def _emit_auto_execute_terminal(
            *,
            user_question: str,
            payload: dict[str, Any],
            debug_tail: str,
        ) -> None:
            nonlocal pending_ui_selection
            answer_raw = str(payload.get("answer", "") or "")
            answer_text, parsed_payload = _extract_structured_answer(answer_raw)
            payload_sources = payload.get("sources", []) if isinstance(payload.get("sources"), list) else []
            payload_trace = payload.get("trace", []) if isinstance(payload.get("trace"), list) else []
            payload_warnings = payload.get("warnings", []) if isinstance(payload.get("warnings"), list) else []
            ui_blocks = _effective_ui_blocks(answer_text, parsed_payload if isinstance(parsed_payload, dict) else None)
            rendered_dynamic = _render_dynamic_blocks(
                console,
                ui_blocks,
                diagram_render_images=diagram_render_images,
                diagram_output_dir=resolved_diagram_output_dir,
                diagram_format=diagram_format,
                diagram_open_artifact=diagram_open,
                diagram_timeout_seconds=diagram_timeout_seconds,
                project_root=root,
            )
            selection_block = _pending_ui_selection_from_blocks(ui_blocks)
            if selection_block is not None:
                pending_ui_selection = selection_block

            warnings_merged = [str(item).strip() for item in payload_warnings if str(item).strip()]
            if isinstance(parsed_payload, dict):
                warnings_merged = _merge_warnings(warnings_merged, parsed_payload)
            auto_trace = _coerce_trace_items(payload_trace)
            changed_files = [str(item) for item in payload.get("changed_files", []) if str(item).strip()]
            if execution_profile == "full-auto":
                answer_text = _sanitize_full_auto_answer_text(
                    answer_text,
                    changed_files_count=len(changed_files),
                    terminal_reason=str(payload.get("terminal_reason", "") or ""),
                )
            turn_record = ChatTurnTelemetry(
                turn_index=len(session_turns) + 1,
                timestamp=_now_iso(),
                question=user_question,
                answer_text=answer_text,
                sources=payload_sources if isinstance(payload_sources, list) else [],
                warnings=warnings_merged,
                trace=auto_trace,
                tool_steps_total=len(auto_trace),
                decisions=_extract_decisions(
                    answer_text=answer_text,
                    warnings=warnings_merged,
                    payload=parsed_payload if isinstance(parsed_payload, dict) else None,
                ),
                changed_files=changed_files,
                has_diff=bool(changed_files),
                coding_state={
                    "plan": payload.get("plan"),
                    "progress": {
                        "phase": "answer",
                        "why": payload.get("terminal_reason", ""),
                    },
                    "checklist": None,
                    "next_step": payload.get("terminal_reason", ""),
                    "flow_id": active_flow_id,
                    "duplicate_request_skips": int(payload.get("duplicate_request_skips", 0) or 0),
                    "duplicate_semantic_search_skips": int(
                        payload.get("duplicate_semantic_search_skips", 0) or 0
                    ),
                    "request_retry_attempts": int(payload.get("request_retry_attempts", 0) or 0),
                    "request_retry_exhausted": int(payload.get("request_retry_exhausted", 0) or 0),
                    "edit_retry_mode_activations": int(payload.get("edit_retry_mode_activations", 0) or 0),
                    "persisted_fingerprint_counts": (
                        dict(payload.get("persisted_fingerprint_counts", {}))
                        if isinstance(payload.get("persisted_fingerprint_counts"), dict)
                        else {}
                    ),
                },
            )
            session_turns.append(turn_record)
            _ = rendered_dynamic
            console.print("\n[bold]Answer[/bold]")
            console.print(Markdown(answer_text) if answer_text else "[dim](no answer text)[/dim]")
            if warnings_merged:
                warning_lines = "\n".join(f"- {w}" for w in warnings_merged[:12])
                console.print(Panel(warning_lines, title="Warnings", border_style="yellow"))
            if debug_tail:
                console.print("\n[bold]Debug tail[/bold]\n" + debug_tail)
            _log_chat_turn(
                run_logger,
                turn=turn_record,
                mode="tools-manager-auto-exec",
                dir_mode=dir_mode,
                coding_agent=bool(coding_agent_instance is not None),
                flow_id=active_flow_id,
                render_mode=str(payload.get("render_mode", "default") or "default"),
                fallback_reason=str(payload.get("fallback_reason", "") or ""),
                planning_mode=planning_mode,
                planning_question_source=planning_question_source,
                planning_question_index=planning_questions_asked_count,
                auto_execute_plan=True,
                auto_execute_passes=int(payload.get("passes", 0) or 0),
                auto_execute_terminal_reason=str(payload.get("terminal_reason", "") or ""),
                toolsmanager_requests_count=int(payload.get("toolsmanager_requests_count", 0) or 0),
                auto_execute_pass_logs=payload.get("pass_logs", []) if isinstance(payload.get("pass_logs"), list) else [],
                planner_decisions=payload.get("planner_decisions", []) if isinstance(payload.get("planner_decisions"), list) else [],
                prechecklist_source=str(payload.get("prechecklist_source", "") or ""),
                prechecklist_steps_count=(
                    len(payload.get("prechecklist", {}).get("steps", []))
                    if isinstance(payload.get("prechecklist"), dict)
                    and isinstance(payload.get("prechecklist", {}).get("steps"), list)
                    else 0
                ),
                prechecklist_warning=str(payload.get("prechecklist_warning", "") or ""),
                tool_execution_backend=str(payload.get("execution_backend", "") or ""),
                tool_execution_run_id=str(payload.get("execution_run_id", "") or ""),
                tool_execution_duration_ms=float(payload.get("execution_duration_ms", 0.0) or 0.0),
                tool_execution_requests_ok=int(payload.get("execution_requests_ok", 0) or 0),
                tool_execution_requests_failed=int(payload.get("execution_requests_failed", 0) or 0),
                full_auto_resume_cycles=int(payload.get("full_auto_resume_cycles", 0) or 0),
                full_auto_passes_total=int(payload.get("full_auto_passes_total", 0) or 0),
                full_auto_pass_checkpoints_emitted=int(payload.get("full_auto_pass_checkpoints_emitted", 0) or 0),
                resumed_from_pass_cap=bool(payload.get("resumed_from_pass_cap", False)),
                multiline_input=multiline_input,
                multiline_terminator=multiline_terminator,
            )

        while True:
            try:
                question = _read_chat_input(
                    console,
                    prompt="💬 » ",
                    multiline_enabled=multiline_input,
                    multiline_terminator=multiline_terminator,
                )
            except (EOFError, KeyboardInterrupt):
                console.print("\nExiting chat.")
                logger.info("Chat session ended by user interrupt/EOF")
                break

            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                console.print("Goodbye!")
                logger.info("Chat session ended by user command", extra={"command": question.lower()})
                break

            if pending_ui_selection is not None:
                if execution_profile == "full-auto":
                    option = _auto_select_ui_option(pending_ui_selection)
                    selection_id = str(pending_ui_selection.get("id", "selection") or "selection")
                    if option is not None:
                        option_id = str(option.get("id", "") or "").replace('"', '\\"')
                        value = str(option.get("value", option_id) or "").replace('"', '\\"')
                        question = (
                            f'User selected "{option_id}" for selection "{selection_id}" '
                            f'(value="{value}") in full-auto mode. Continue accordingly.'
                        )
                        logger.info(
                            "Full-auto selection resolution",
                            extra={"selection_id": selection_id, "option_id": option_id},
                        )
                        pending_ui_selection = None
                    else:
                        logger.info(
                            "Full-auto selection dropped (no options)",
                            extra={"selection_id": selection_id},
                        )
                        pending_ui_selection = None
                if pending_ui_selection is None:
                    pass
                else:
                    selection_kind, selection_payload = _resolve_ui_selection_input(pending_ui_selection, question)
                    selection_id = str(pending_ui_selection.get("id", "selection") or "selection")

                    if selection_kind == "invalid":
                        console.print("[yellow]Invalid selection. Choose by number, option id, or label.[/yellow]")
                        _render_selection_block(console, pending_ui_selection)
                        continue

                    if selection_kind == "free_text":
                        raw_text = str((selection_payload or {}).get("text", "") or "").replace('"', '\\"')
                        question = (
                            f'User provided free-text response "{raw_text}" '
                            f'for selection "{selection_id}". Continue accordingly.'
                        )
                        pending_ui_selection = None
                    else:
                        option = selection_payload or {}
                        option_id = str(option.get("id", "") or "").replace('"', '\\"')
                        value = str(option.get("value", option_id) or "").replace('"', '\\"')
                        question = (
                            f'User selected "{option_id}" for selection "{selection_id}" '
                            f'(value="{value}"). Continue accordingly.'
                        )
                        pending_ui_selection = None

            if coding_agent_instance is not None and question.startswith("/flow"):
                parts = question.strip().split()
                action = parts[1].lower() if len(parts) > 1 else "show"
                if action == "show":
                    summary = coding_agent_instance.flow_summary(active_flow_id)
                    if not summary:
                        console.print("[yellow]No active coding flow.[/yellow]")
                        continue
                    _render_flow_summary(
                        console,
                        summary,
                        include_checklist=False,
                        include_transitions=False,
                        include_recent_turns=False,
                    )
                    continue
                if action == "checklist":
                    summary = coding_agent_instance.flow_summary(active_flow_id)
                    checklist = summary.get("checklist") if isinstance(summary, dict) else None
                    if not isinstance(checklist, dict):
                        console.print("[yellow]No checklist stored for the active flow.[/yellow]")
                        continue
                    _render_flow_checklist(console, checklist)
                    continue
                if action == "checkpoint":
                    checkpointed = coding_agent_instance.checkpoint_flow(active_flow_id)
                    if checkpointed:
                        console.print(f"[green]Checkpoint saved for flow {checkpointed}.[/green]")
                    else:
                        console.print("[yellow]No active flow to checkpoint.[/yellow]")
                    continue
                if action == "reset":
                    reset_id = coding_agent_instance.reset_flow(active_flow_id)
                    active_flow_id = None
                    pending_conflict_question = None
                    if reset_id:
                        console.print(f"[green]Flow reset: {reset_id}[/green]")
                    else:
                        console.print("[yellow]No active flow to reset.[/yellow]")
                    continue
                console.print(
                    "[yellow]Unknown /flow command. Use /flow show, /flow checklist, /flow checkpoint, or /flow reset.[/yellow]"
                )
                continue

            plan_trigger_request = _looks_like_plan_trigger_request(question)
            force_auto_execute_edit = bool(
                execution_profile == "full-auto"
                and coding_agent_instance is not None
                and auto_execute_plan
                and _looks_like_edit_request(question)
            )

            if (
                coding_agent_instance is not None
                and auto_execute_plan
                and hasattr(coding_agent_instance, "generate_auto_execute")
                and (plan_trigger_request or force_auto_execute_edit)
            ):
                pending_prechecklist = None
                pending_prechecklist_source = ""
                pending_prechecklist_warning = ""
                if hasattr(coding_agent_instance, "preview_execution_checklist"):
                    try:
                        preview_payload = coding_agent_instance.preview_execution_checklist(
                            question,
                            flow_id=active_flow_id,
                        )
                    except Exception as exc:
                        _log_exception("coding_agent.preview_execution_checklist", exc)
                        preview_payload = {}
                    if isinstance(preview_payload, dict):
                        flow_id_from_preview = preview_payload.get("flow_id")
                        if isinstance(flow_id_from_preview, str) and flow_id_from_preview.strip():
                            active_flow_id = flow_id_from_preview.strip()
                        preview = preview_payload.get("prechecklist")
                        if isinstance(preview, dict):
                            pending_prechecklist = preview
                            pending_prechecklist_source = str(preview_payload.get("prechecklist_source", "") or "")
                            pending_prechecklist_warning = str(preview_payload.get("prechecklist_warning", "") or "")
                target_flow = active_flow_id or coding_agent_instance.get_active_flow_id()
                if isinstance(target_flow, str) and target_flow.strip():
                    active_flow_id = target_flow.strip()

            if planning_mode:
                if planning_request is None:
                    planning_request = question
                    planning_answers = []
                    planning_questions = []
                    planning_questions_asked_count = 0
                    try:
                        llm_question = _generate_planning_question_llm(
                            ask_service=ask_service,
                            planning_request=planning_request,
                            prior_questions=planning_questions,
                            prior_answers=planning_answers,
                            asked_count=0,
                            max_questions=planning_question_limit,
                        )
                        planning_question_source = "llm"
                    except Exception as exc:
                        logger.warning("Planning question generation failed; using static fallback: %s", exc)
                        llm_question = _planning_questions(planning_question_limit)[0]
                        planning_question_source = "fallback_static"
                    planning_questions.append(llm_question)
                    console.print(
                        f"[cyan]Planning request:[/cyan] {planning_request}\n"
                        f"[bold]Planning question 1/{planning_question_limit}[/bold]\n"
                        f"{llm_question}"
                    )
                    continue

                planning_answers.append(question)
                if len(planning_answers) < planning_question_limit:
                    asked_count = len(planning_answers)
                    try:
                        llm_question = _generate_planning_question_llm(
                            ask_service=ask_service,
                            planning_request=planning_request,
                            prior_questions=planning_questions,
                            prior_answers=planning_answers,
                            asked_count=asked_count,
                            max_questions=planning_question_limit,
                        )
                        planning_question_source = "llm"
                    except Exception as exc:
                        logger.warning("Planning question generation failed; using static fallback: %s", exc)
                        llm_question = _planning_questions(planning_question_limit)[asked_count]
                        planning_question_source = "fallback_static"
                    planning_questions.append(llm_question)
                    console.print(
                        f"[cyan]Planning request:[/cyan] {planning_request}\n"
                        f"[bold]Planning question {asked_count + 1}/{planning_question_limit}[/bold]\n"
                        f"{llm_question}"
                    )
                    continue

                question = _build_planning_instruction(
                    planning_request,
                    planning_answers,
                    planning_question_limit,
                    questions=planning_questions,
                )
                logger.info(
                    "Planning Q&A complete; generating plan response",
                    extra={
                        "planning_request": planning_request,
                        "answers_count": len(planning_answers),
                        "question_source": planning_question_source,
                    },
                )
                planning_questions_asked_count = len(planning_answers)
                planning_request = None
                planning_answers = []
                planning_questions = []
                if agent_tools and auto_execute_plan:
                    auto_payload, auto_debug_tail = _run_auto_execute_pipeline_with_resume(question)
                    _emit_auto_execute_terminal(
                        user_question=question,
                        payload=auto_payload,
                        debug_tail=auto_debug_tail,
                    )
                    continue
                console.print("[cyan]Generating decision-complete plan...[/cyan]")

            logger.info("Chat question received", extra={"question": question, "dir_mode": dir_mode, "agent_tools": agent_tools})

            if (
                _looks_like_edit_request(question)
                and coding_agent_instance is None
                and not (agent_tools and auto_execute_plan and tool_worker_process and _looks_like_plan_trigger_request(question))
            ):
                console.print(
                    "[yellow]This chat session is read-only for file edits.[/yellow] "
                    "Re-run with [bold]--agent-tools --coding-agent[/bold] to allow write_file/apply_patch."
                )
                continue

            if (
                coding_agent_instance is not None
                and coding_memory
                and _looks_like_edit_request(question)
                and not plan_trigger_request
            ):
                if pending_conflict_question is not None:
                    choice = question.strip().lower()
                    if choice in {"continue", "c", "1"}:
                        question = pending_conflict_question
                    elif choice in {"new", "n", "2"}:
                        active_flow_id = None
                        question = pending_conflict_question
                    else:
                        console.print("[yellow]Reply 'continue' or 'new'.[/yellow]")
                        continue
                    pending_conflict_question = None
                elif active_flow_id and coding_agent_instance.is_conflicting_request(question, active_flow_id):
                    if execution_profile == "full-auto":
                        logger.info(
                            "Full-auto flow conflict auto-continued",
                            extra={"flow_id": active_flow_id, "question": question},
                        )
                    else:
                        pending_conflict_question = question
                        console.print(
                            "[yellow]This request appears to diverge from the active flow.[/yellow] "
                            "Type [bold]continue[/bold] to keep current flow or [bold]new[/bold] to start a new flow."
                        )
                        continue

            if (
                coding_agent_instance is None
                and agent_tools
                and auto_execute_plan
                and _looks_like_plan_trigger_request(question)
            ):
                auto_payload, auto_debug_tail = _run_auto_execute_pipeline_with_resume(
                    question,
                    render_progress=False,
                )
                _emit_auto_execute_terminal(
                    user_question=question,
                    payload=auto_payload,
                    debug_tail=auto_debug_tail,
                )
                continue

            # ==========================================================
            # ✅ CODING AGENT PATH (classic + dir-mode supported)
            # ==========================================================
            if coding_agent_instance is not None:
                cb = RichToolCallbackHandler(show_inputs=True)
                execute_plan_now = bool(auto_execute_plan and (plan_trigger_request or force_auto_execute_edit))
                request_for_generation = question
                turn_full_auto_resume_cycles = 0
                turn_full_auto_passes_total = 0
                turn_full_auto_pass_checkpoints_emitted = 0
                turn_resumed_from_pass_cap = False

                try:
                    if dir_mode:
                        index_dirs = dir_mode_index_dirs
                        if not index_dirs:
                            console.print("[red]No indexes available in dir-mode.[/red]")
                            continue

                        def _call(callbacks: list[BaseCallbackHandler]):
                            if execute_plan_now and hasattr(coding_agent_instance, "generate_auto_execute"):
                                return coding_agent_instance.generate_auto_execute(
                                    request_for_generation,
                                    index_dirs=index_dirs,
                                    k=resolved_k,
                                    max_steps=coding_agent_max_steps,
                                    timeout_seconds=min(max(agent_timeout_seconds, 60), 600),
                                    pass_cap=auto_execute_max_passes,
                                    callbacks=callbacks,
                                    flow_id=active_flow_id,
                                    prechecklist_payload=(
                                        {
                                            "flow_id": active_flow_id,
                                            "prechecklist": pending_prechecklist,
                                            "prechecklist_source": pending_prechecklist_source,
                                            "prechecklist_warning": pending_prechecklist_warning,
                                        }
                                        if isinstance(pending_prechecklist, dict)
                                        else None
                                    ),
                                )
                            return coding_agent_instance.generate_dir_mode(
                                request_for_generation,
                                index_dirs=index_dirs,
                                k=resolved_k,
                                max_steps=coding_agent_max_steps,
                                timeout_seconds=min(max(agent_timeout_seconds, 60), 600),
                                callbacks=callbacks,
                                flow_id=active_flow_id,
                            )
                    else:
                        assert resolved_index_dir is not None

                        def _call(callbacks: list[BaseCallbackHandler]):
                            if execute_plan_now and hasattr(coding_agent_instance, "generate_auto_execute"):
                                return coding_agent_instance.generate_auto_execute(
                                    request_for_generation,
                                    index_dir=resolved_index_dir,
                                    k=resolved_k,
                                    max_steps=coding_agent_max_steps,
                                    timeout_seconds=min(max(agent_timeout_seconds, 60), 600),
                                    pass_cap=auto_execute_max_passes,
                                    callbacks=callbacks,
                                    flow_id=active_flow_id,
                                    prechecklist_payload=(
                                        {
                                            "flow_id": active_flow_id,
                                            "prechecklist": pending_prechecklist,
                                            "prechecklist_source": pending_prechecklist_source,
                                            "prechecklist_warning": pending_prechecklist_warning,
                                        }
                                        if isinstance(pending_prechecklist, dict)
                                        else None
                                    ),
                                )
                            return coding_agent_instance.generate(
                                request_for_generation,
                                index_dir=resolved_index_dir,
                                k=resolved_k,
                                max_steps=coding_agent_max_steps,
                                timeout_seconds=min(max(agent_timeout_seconds, 60), 600),
                                callbacks=callbacks,
                                flow_id=active_flow_id,
                            )

                    while True:
                        result, debug_tail = _run_with_live_buffer(
                            console,
                            spinner_text="Coding…",
                            fn=_call,
                            callbacks=[cb],
                        )
                        if not (
                            execution_profile == "full-auto"
                            and execute_plan_now
                            and isinstance(result, dict)
                        ):
                            break
                        turn_full_auto_passes_total += _ingest_full_auto_pass_payload(result)
                        terminal_reason = str((result or {}).get("auto_execute_terminal_reason", "") or "").strip().lower()
                        flow_from_cycle = (result or {}).get("flow_id")
                        if isinstance(flow_from_cycle, str) and flow_from_cycle.strip():
                            active_flow_id = flow_from_cycle.strip()
                        if terminal_reason != "pass_cap_reached":
                            break
                        turn_resumed_from_pass_cap = True
                        turn_full_auto_resume_cycles += 1
                        turn_full_auto_pass_checkpoints_emitted += _emit_full_auto_pass_checkpoints(
                            resume_cycles=turn_full_auto_resume_cycles
                        )
                        request_for_generation = _build_full_auto_resume_request(
                            original_question=question,
                            resume_cycle=turn_full_auto_resume_cycles,
                            terminal_reason=terminal_reason,
                        )
                        continue

                except ToolWorkerProcessError as exc:
                    _log_exception("coding_agent.generate.worker", exc)
                    if exc.code == "tools_only_violation":
                        console.print(
                            "[yellow]Tools-only policy blocked this request:[/yellow] "
                            "no successful tool calls were executed. "
                            "Ask for specific file/tool actions and retry."
                        )
                        continue
                    console.print(
                        "[red]Coding agent worker failed.[/red] "
                        "Retrying this turn read-only is recommended."
                    )
                    continue
                except Exception as exc:
                    _log_exception("coding_agent.generate", exc)
                    console.print("[red]Coding agent failed.[/red]")
                    continue

                # result schema expected from CodingAgent
                answer = str((result or {}).get("answer", "") or "")
                changed = (result or {}).get("changed_files", []) or []
                diff = str((result or {}).get("diff", "") or "")
                warns = (result or {}).get("warnings", []) or []
                result_flow_id = (result or {}).get("flow_id")
                if isinstance(result_flow_id, str) and result_flow_id.strip():
                    active_flow_id = result_flow_id.strip()
                if execute_plan_now and isinstance(result, dict):
                    if isinstance(pending_prechecklist, dict) and not isinstance(result.get("prechecklist"), dict):
                        result["prechecklist"] = pending_prechecklist
                    if pending_prechecklist_source and not str(result.get("prechecklist_source", "")).strip():
                        result["prechecklist_source"] = pending_prechecklist_source
                    if pending_prechecklist_warning and not str(result.get("prechecklist_warning", "")).strip():
                        result["prechecklist_warning"] = pending_prechecklist_warning
                    result["full_auto_resume_cycles"] = int(turn_full_auto_resume_cycles)
                    result["full_auto_passes_total"] = int(turn_full_auto_passes_total)
                    result["full_auto_pass_checkpoints_emitted"] = int(turn_full_auto_pass_checkpoints_emitted)
                    result["resumed_from_pass_cap"] = bool(turn_resumed_from_pass_cap)
                answer_text, parsed_payload = _extract_structured_answer(answer)
                if execution_profile == "full-auto" and execute_plan_now:
                    answer_text = _sanitize_full_auto_answer_text(
                        answer_text,
                        changed_files_count=len([str(item) for item in changed if str(item).strip()]),
                        terminal_reason=(
                            str((result or {}).get("auto_execute_terminal_reason", "") or "")
                            if isinstance(result, dict)
                            else ""
                        ),
                    )
                payload_sources = []
                payload_warnings = []
                payload_trace = []
                payload_ui_blocks: list[dict[str, Any]] = []
                if isinstance(parsed_payload, dict):
                    if isinstance(parsed_payload.get("sources"), list):
                        payload_sources = parsed_payload["sources"]
                    if isinstance(parsed_payload.get("warnings"), list):
                        payload_warnings = [str(w).strip() for w in parsed_payload["warnings"] if str(w).strip()]
                    if isinstance(parsed_payload.get("trace"), list):
                        payload_trace = parsed_payload["trace"]
                    payload_ui_blocks = _effective_ui_blocks(answer_text, parsed_payload)
                else:
                    payload_ui_blocks = _effective_ui_blocks(answer_text, None)
                merged_warns = list(warns)
                for warning in payload_warnings:
                    if warning not in merged_warns:
                        merged_warns.append(warning)
                # CodingAgent outputs actions_taken, not trace
                result_actions = (result or {}).get("actions_taken", []) or []
                result_trace = (result or {}).get("trace", []) or []  # legacy/compat
                effective_trace = result_actions or result_trace or payload_trace
                raw_actions_total = (result or {}).get("actions_taken_total")
                if isinstance(raw_actions_total, int):
                    actions_total = raw_actions_total
                else:
                    actions_total = len(effective_trace)
                if isinstance(result, dict):
                    existing_actions = result.get("actions_taken")
                    if effective_trace:
                        result["actions_taken"] = effective_trace
                    elif isinstance(existing_actions, list):
                        result["actions_taken"] = existing_actions
                    else:
                        result["actions_taken"] = []
                    result["actions_taken_total"] = actions_total
                    result["actions_taken_truncated"] = actions_total > len(result.get("actions_taken", []))
                    result["warnings"] = merged_warns
                render_mode = (
                    str((result or {}).get("render_mode", "")).strip().lower()
                    if isinstance(result, dict)
                    else ""
                )
                fallback_reason = (
                    str((result or {}).get("fallback_reason", "")).strip().lower()
                    if isinstance(result, dict)
                    else ""
                )
                answer_only_fallback = render_mode == "answer_only" and fallback_reason == "tools_only_violation"
                answer_only_auto_execute = bool(execute_plan_now)
                edit_completed = bool(changed) or bool(diff.strip())
                answer_only_no_edit = (not answer_only_fallback) and (not edit_completed) and (not answer_only_auto_execute)

                rendered_dynamic: dict[str, bool] = {}
                if not answer_only_fallback:
                    rendered_dynamic = _render_dynamic_blocks(
                        console,
                        payload_ui_blocks,
                        diagram_render_images=diagram_render_images,
                        diagram_output_dir=resolved_diagram_output_dir,
                        diagram_format=diagram_format,
                        diagram_open_artifact=diagram_open,
                        diagram_timeout_seconds=diagram_timeout_seconds,
                        project_root=root,
                    )
                selection_block = _pending_ui_selection_from_blocks(payload_ui_blocks)
                if selection_block is not None:
                    pending_ui_selection = selection_block
                turn_record = ChatTurnTelemetry(
                    turn_index=len(session_turns) + 1,
                    timestamp=_now_iso(),
                    question=question,
                    answer_text=answer_text,
                    sources=list(payload_sources),
                    warnings=list(merged_warns),
                    trace=_coerce_trace_items(effective_trace),
                    tool_steps_total=actions_total,
                    decisions=_extract_decisions(
                        answer_text=answer_text,
                        warnings=merged_warns,
                        payload=parsed_payload if isinstance(parsed_payload, dict) else None,
                        result_payload=result if isinstance(result, dict) else None,
                    ),
                    changed_files=[str(item) for item in changed if str(item).strip()],
                    has_diff=bool(diff.strip()),
                    coding_state={
                        "plan": (result or {}).get("plan") if isinstance(result, dict) else None,
                        "progress": (result or {}).get("progress") if isinstance(result, dict) else None,
                        "checklist": (result or {}).get("checklist") if isinstance(result, dict) else None,
                        "next_step": (result or {}).get("next_step") if isinstance(result, dict) else None,
                        "flow_id": active_flow_id,
                        "duplicate_request_skips": (
                            int((result or {}).get("duplicate_request_skips", 0) or 0)
                            if isinstance(result, dict)
                            else 0
                        ),
                        "duplicate_semantic_search_skips": (
                            int((result or {}).get("duplicate_semantic_search_skips", 0) or 0)
                            if isinstance(result, dict)
                            else 0
                        ),
                        "request_retry_attempts": (
                            int((result or {}).get("request_retry_attempts", 0) or 0)
                            if isinstance(result, dict)
                            else 0
                        ),
                        "request_retry_exhausted": (
                            int((result or {}).get("request_retry_exhausted", 0) or 0)
                            if isinstance(result, dict)
                            else 0
                        ),
                        "edit_retry_mode_activations": (
                            int((result or {}).get("edit_retry_mode_activations", 0) or 0)
                            if isinstance(result, dict)
                            else 0
                        ),
                        "persisted_fingerprint_counts": (
                            dict((result or {}).get("persisted_fingerprint_counts", {}))
                            if isinstance(result, dict)
                            and isinstance((result or {}).get("persisted_fingerprint_counts"), dict)
                            else {}
                        ),
                    },
                )
                session_turns.append(turn_record)
                if answer_only_fallback or answer_only_no_edit or answer_only_auto_execute:
                    console.print("\n[bold]Answer[/bold]")
                    if answer_text:
                        console.print(Markdown(answer_text))
                    else:
                        console.print("[dim](no answer text)[/dim]")
                else:
                    _render_turn_transparency(
                        console,
                        turn=turn_record,
                        history=session_turns,
                    )
                _log_chat_turn(
                    run_logger,
                    turn=turn_record,
                    mode="coding-agent",
                    dir_mode=dir_mode,
                    coding_agent=True,
                    flow_id=active_flow_id,
                    render_mode=render_mode,
                    fallback_reason=fallback_reason,
                    planning_mode=planning_mode,
                    planning_question_source=planning_question_source,
                    planning_question_index=planning_questions_asked_count,
                    auto_execute_plan=execute_plan_now,
                    auto_execute_passes=int((result or {}).get("auto_execute_passes", 0) or 0) if isinstance(result, dict) else 0,
                    auto_execute_terminal_reason=str((result or {}).get("auto_execute_terminal_reason", "") or "") if isinstance(result, dict) else "",
                    toolsmanager_requests_count=int((result or {}).get("toolsmanager_requests_count", 0) or 0) if isinstance(result, dict) else 0,
                    auto_execute_pass_logs=(result or {}).get("pass_logs", []) if isinstance(result, dict) else [],
                    planner_decisions=(result or {}).get("planner_decisions", []) if isinstance(result, dict) else [],
                    prechecklist_source=(
                        str((result or {}).get("prechecklist_source", "") or "")
                        if isinstance(result, dict)
                        else pending_prechecklist_source
                    ),
                    prechecklist_steps_count=(
                        len((result or {}).get("prechecklist", {}).get("steps", []))
                        if isinstance(result, dict)
                        and isinstance((result or {}).get("prechecklist"), dict)
                        and isinstance((result or {}).get("prechecklist", {}).get("steps"), list)
                        else (len(pending_prechecklist.get("steps", [])) if isinstance(pending_prechecklist, dict) and isinstance(pending_prechecklist.get("steps"), list) else 0)
                    ),
                    prechecklist_warning=(
                        str((result or {}).get("prechecklist_warning", "") or "")
                        if isinstance(result, dict)
                        else pending_prechecklist_warning
                    ),
                    tool_execution_backend=(
                        str((result or {}).get("tool_execution_backend", "") or "")
                        if isinstance(result, dict)
                        else ""
                    ),
                    tool_execution_run_id=(
                        str((result or {}).get("tool_execution_run_id", "") or "")
                        if isinstance(result, dict)
                        else ""
                    ),
                    tool_execution_duration_ms=(
                        float((result or {}).get("tool_execution_duration_ms", 0.0) or 0.0)
                        if isinstance(result, dict)
                        else 0.0
                    ),
                    tool_execution_requests_ok=(
                        int((result or {}).get("tool_execution_requests_ok", 0) or 0)
                        if isinstance(result, dict)
                        else 0
                    ),
                    tool_execution_requests_failed=(
                        int((result or {}).get("tool_execution_requests_failed", 0) or 0)
                        if isinstance(result, dict)
                        else 0
                    ),
                    full_auto_resume_cycles=(
                        int((result or {}).get("full_auto_resume_cycles", 0) or 0)
                        if isinstance(result, dict)
                        else 0
                    ),
                    full_auto_passes_total=(
                        int((result or {}).get("full_auto_passes_total", 0) or 0)
                        if isinstance(result, dict)
                        else 0
                    ),
                    full_auto_pass_checkpoints_emitted=(
                        int((result or {}).get("full_auto_pass_checkpoints_emitted", 0) or 0)
                        if isinstance(result, dict)
                        else 0
                    ),
                    resumed_from_pass_cap=(
                        bool((result or {}).get("resumed_from_pass_cap", False))
                        if isinstance(result, dict)
                        else False
                    ),
                    multiline_input=multiline_input,
                    multiline_terminator=multiline_terminator,
                )
                pending_prechecklist = None
                pending_prechecklist_source = ""
                pending_prechecklist_warning = ""
                if not (answer_only_fallback or answer_only_no_edit or answer_only_auto_execute):
                    _render_coding_sections(
                        console,
                        result if isinstance(result, dict) else {},
                        rendered_dynamic=rendered_dynamic,
                        show_actions=False,
                    )
                    if payload_sources:
                        _render_answer_sections(
                            console,
                            answer="",
                            title="Sources",
                            sources=payload_sources,
                            warnings=[],
                            trace=[],
                            show_trace=False,
                        )
                    if debug_tail:
                        console.print("\n[bold]Debug tail[/bold]\n" + debug_tail)

                # Optional: if you want quick diff visibility without full diff spam:
                # console.print("\n[dim]Tip: run with your own :diff command if you add history later.[/dim]")

                continue
    finally:
        if tool_worker_client is not None:
            tool_worker_client.stop()
        if tmp_root is not None:
            tmp_root.cleanup()
        if tmp_base is not None:
            tmp_base.cleanup()
