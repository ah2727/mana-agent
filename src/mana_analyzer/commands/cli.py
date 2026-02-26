from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import hashlib

import typer
from langchain_community.embeddings import OpenAIEmbeddings
from rich.console import Console

from mana_analyzer.analysis.checks import PythonStaticAnalyzer
from mana_analyzer.analysis.chunker import CodeChunker
from mana_analyzer.config.settings import Settings, default_index_dir
from mana_analyzer.llm.ask_agent import AskAgent
from mana_analyzer.llm.analyze_chain import AnalyzeChain
from mana_analyzer.llm.qna_chain import QnAChain
from mana_analyzer.llm.repo_chain import RepositoryMultiChain
from mana_analyzer.llm.coding_agent import CodingAgent  # ✅ NEW
from mana_analyzer.parsers.multi_parser import MultiLanguageParser
from mana_analyzer.services.analyze_service import AnalyzeService
from mana_analyzer.services.ask_service import AskService
from mana_analyzer.services.dependency_service import DependencyService
from mana_analyzer.services.describe_service import DescribeService
from mana_analyzer.services.index_service import IndexService
from mana_analyzer.services.llm_analyze_service import LlmAnalyzeService
from mana_analyzer.services.search_service import SearchService
from mana_analyzer.services.structure_service import StructureService
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
import time

app = typer.Typer(help="mana-analyzer CLI")
console = Console()
logger = logging.getLogger(__name__)
OUTPUT_DIR: Path | None = None


class RichToolCallbackHandler(BaseCallbackHandler):
    """Live-updates a Rich spinner when tools start/end."""

    def __init__(self, live: Live, *, show_inputs: bool = True) -> None:
        self.live = live
        self.show_inputs = show_inputs
        self._tool: str | None = None
        self._t0: float = 0.0

    def on_tool_start(self, serialized, input_str: str, **kwargs) -> None:
        name = (serialized or {}).get("name") or "tool"
        self._tool = str(name)
        self._t0 = time.time()
        msg = f"Using tool: {self._tool}"
        if self.show_inputs and input_str:
            inp = input_str.strip().replace("\n", " ")
            if len(inp) > 160:
                inp = inp[:160] + "…"
            msg += f"\n↳ {inp}"
        self.live.update(Text(msg))

    def on_tool_end(self, output: str, **kwargs) -> None:
        tool = self._tool or "tool"
        dt = max(0.0, time.time() - self._t0)
        self._tool = None
        self.live.update(Text(f"Finished: {tool} ({dt:0.1f}s)"))

    def on_tool_error(self, error: BaseException, **kwargs) -> None:
        tool = self._tool or "tool"
        self._tool = None
        self.live.update(Text(f"Tool error: {tool} - {error}"))


# -----------------------------------------
# "Full logging" helpers (added, non-breaking)
# -----------------------------------------


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _log_call(fn_name: str, **fields: object) -> None:
    try:
        logger.debug("CALL %s", fn_name, extra={**fields, "ts": _now_iso()})
    except Exception:
        logger.debug("CALL %s (extra logging failed)", fn_name)


def _log_return(fn_name: str, **fields: object) -> None:
    try:
        logger.debug("RETURN %s", fn_name, extra={**fields, "ts": _now_iso()})
    except Exception:
        logger.debug("RETURN %s (extra logging failed)", fn_name)


def _log_exception(fn_name: str, exc: Exception, **fields: object) -> None:
    try:
        logger.exception("EXCEPTION %s: %s", fn_name, exc, extra={**fields, "ts": _now_iso()})
    except Exception:
        logger.exception("EXCEPTION %s: %s", fn_name, exc)


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
    log_dir: str | None = typer.Option(None, "--log-dir", help="Directory for application log files."),
    output_dir: str | None = typer.Option(None, "--output-dir", help="Directory for saving command output logs."),
) -> None:
    global OUTPUT_DIR
    OUTPUT_DIR = Path(output_dir).resolve() if output_dir else None
    log_file = setup_logging(verbose=verbose, log_dir=log_dir)
    logger.debug(
        "CLI initialized",
        extra={"verbose": verbose, "log_file": str(log_file), "output_dir": str(OUTPUT_DIR) if OUTPUT_DIR else None},
    )



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
        payload["summarization"] = summary_payload
        markdown = (
            structure_service.render_markdown(report).rstrip()
            + "\n\n"
            + _render_repository_summary_markdown(summary_payload)
        )
        if output_format not in {"json", "markdown", "both"}:
            raise typer.BadParameter("--output-format must be one of: json, markdown, both")
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
    agent_tools: bool = typer.Option(False, "--agent-tools"),
    agent_max_steps: int = typer.Option(6, "--agent-max-steps"),
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
                    max_steps=agent_max_steps,
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
                    max_steps=agent_max_steps,
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
        help="Automatically create missing subproject indexes in dir-mode.",
    ),
    agent_tools: bool = typer.Option(
        False,
        "--agent-tools",
        help="Enable tool-aware answering (calls specialized tools).",
    ),
    coding_agent: bool = typer.Option(
        False,
        "--coding-agent",
        help="Enable repo-modifying coding agent (write_file/apply_patch) inside chat.",
    ),
    agent_max_steps: int = typer.Option(6, "--agent-max-steps"),
    agent_timeout_seconds: int = typer.Option(30, "--agent-timeout-seconds"),
    as_json: bool = typer.Option(False, "--json", help="Emit responses as JSON objects."),
) -> None:
    output_file = _resolve_output_file()

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
            "agent_max_steps": agent_max_steps,
            "agent_timeout_seconds": agent_timeout_seconds,
            "as_json": as_json,
            "ephemeral_index": ephemeral_index,
        },
    )

    settings = Settings()
    resolved_k = k or settings.default_top_k

    if dir_mode:
        root = Path(root_dir).resolve() if root_dir else Path.cwd().resolve()
    else:
        root = Path.cwd().resolve()
    if root.is_file():
        root = root.parent

    logger.debug("Resolved chat root", extra={"root": str(root)})

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
        agent_max_steps=agent_max_steps,
        agent_timeout_seconds=agent_timeout_seconds,
        max_indexes=max_indexes,
        auto_index_missing=auto_index_missing,
    )

    # ✅ Initialize CodingAgent (only when enabled)
    coding_agent_instance: CodingAgent | None = None
    if coding_agent:
        if not agent_tools:
            raise typer.BadParameter("--coding-agent requires --agent-tools (needs tool loop).")
        if getattr(ask_service, "ask_agent", None) is None:
            raise typer.BadParameter("--coding-agent requires AskService.ask_agent to be configured.")

        # ✅ IMPORTANT: allow_prefixes=None => unrestricted under repo_root
        coding_agent_instance = CodingAgent(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            repo_root=root,
            ask_agent=ask_service.ask_agent,
            allowed_prefixes=None,
        )

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
                f"- coding-agent: {coding_agent}",
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
                f"- coding-agent: {coding_agent}",
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
        if not agent_tools:
            console.print("[yellow]Tip:[/yellow] run with --agent-tools for real tool-driven investigation.")
        if coding_agent:
            console.print("[bold red]Coding agent enabled:[/bold red] this session can modify any files under the repository root.")

        while True:
            try:
                question = input("💬 » ").strip()
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

            logger.info("Chat question received", extra={"question": question, "dir_mode": dir_mode, "agent_tools": agent_tools})


            # ==========================================================
            # ✅ CODING AGENT PATH (classic + dir-mode supported)
            # ==========================================================
            if coding_agent_instance is not None:
                spinner = Spinner("dots", text="Coding…")
                with Live(spinner, console=console, refresh_per_second=12, transient=True) as live:
                    cb = RichToolCallbackHandler(live, show_inputs=True)

                    try:
                        if dir_mode:
                            index_dirs = dir_mode_index_dirs
                            if not index_dirs:
                                console.print("[red]No indexes available in dir-mode (internal CLI selection is empty).[/red]")
                                continue

                            result = coding_agent_instance.generate_dir_mode(
                                question,
                                index_dirs=index_dirs,
                                k=resolved_k,
                                max_steps=min(max(agent_max_steps, 8), 200),
                                timeout_seconds=min(max(agent_timeout_seconds, 60), 600),
                                callbacks=[cb],
                            )
                        else:
                            assert resolved_index_dir is not None
                            result = coding_agent_instance.generate(
                                question,
                                index_dir=resolved_index_dir,
                                k=resolved_k,
                                max_steps=min(max(agent_max_steps, 8), 200),
                                timeout_seconds=min(max(agent_timeout_seconds, 60), 600),
                                callbacks=[cb],
                            )
                    except Exception as exc:
                        _log_exception("coding_agent.generate", exc)
                        raise

                if as_json:
                    console.print_json(json.dumps(result))
                else:
                    console.print(result.get("answer", ""))

                    changed = result.get("changed_files", []) or []
                    if changed:
                        console.print("\n[bold]Changed files:[/bold]")
                        for p in changed:
                            console.print(f"- {p}")
                    else:
                        console.print(
                            "\n[yellow]No files were changed.[/yellow]\n"
                            "Common causes:\n"
                            "- tool loop did not run (agent didn't call write_file/apply_patch)\n"
                            "- write was blocked by allowed prefixes\n"
                            "- patch failed to apply cleanly\n"
                        )

                    diff = result.get("diff", "") or ""
                    if diff:
                        console.print("\n[bold]Diff:[/bold]\n" + diff)

                    sa = result.get("static_analysis", {}) or {}
                    if sa.get("finding_count", 0):
                        console.print("\n[bold yellow]Static analysis findings:[/bold yellow]")
                        for f in sa.get("findings", []) or []:
                            console.print(f"- {f}")

                    warns = result.get("warnings", []) or []
                    if warns:
                        console.print("\nWarnings:\n" + "\n".join(f"- {w}" for w in warns))

                continue
            # ==========================================================
            # ✅ NORMAL CHAT PATH (your existing logic)
            # ==========================================================
            response = None
            attempt_question = question

            for attempt in range(1, max_attempts + 1):
                logger.debug("Chat attempt", extra={"attempt": attempt, "max_attempts": max_attempts})

                try:
                    if agent_tools:
                        spinner = Spinner("dots", text="Thinking…")
                        with Live(spinner, console=console, refresh_per_second=12, transient=True) as live:
                            cb = RichToolCallbackHandler(live, show_inputs=True)

                            try:
                                response = chat_service.ask(attempt_question, callbacks=[cb])  # type: ignore[arg-type]
                            except TypeError:
                                if dir_mode:
                                    response = ask_service.ask_with_tools_dir_mode(  # type: ignore[call-arg]
                                        index_dirs=getattr(chat_service, "index_dirs", []) or [],
                                        question=attempt_question,
                                        k=resolved_k,
                                        max_steps=agent_max_steps,
                                        timeout_seconds=agent_timeout_seconds,
                                        root_dir=root,
                                        callbacks=[cb],
                                    )
                                else:
                                    assert resolved_index_dir is not None
                                    response = ask_service.ask_with_tools(
                                        index_dir=resolved_index_dir,
                                        question=attempt_question,
                                        k=resolved_k,
                                        max_steps=agent_max_steps,
                                        timeout_seconds=agent_timeout_seconds,
                                        callbacks=[cb],
                                    )
                    else:
                        response = chat_service.ask(attempt_question)
                except Exception as exc:
                    _log_exception("chat_service.ask", exc, attempt=attempt)
                    raise

                if response is None:
                    logger.warning("Chat service returned None response", extra={"attempt": attempt})
                    break

                src_count = len(getattr(response, "sources", []) or [])
                guessed = _looks_like_guess(getattr(response, "answer", ""))

                logger.debug(
                    "Chat attempt evaluation",
                    extra={"attempt": attempt, "src_count": src_count, "guessed": guessed},
                )

                if (src_count >= min_sources) and not guessed:
                    break

                if attempt < max_attempts:
                    attempt_question = _tool_first_instruction(question) + f"\n(Attempt {attempt+1}/{max_attempts})"

            if response is None:
                continue

            if as_json:
                console.print_json(json.dumps(response.to_dict()))
            else:
                console.print(response.answer)
                if response.sources:
                    console.print(
                        "\nSources:\n"
                        + "\n".join(f"- {src.file_path}:{src.start_line}-{src.end_line}" for src in response.sources)
                    )
                if getattr(response, "warnings", None):
                    console.print("\nWarnings:\n" + "\n".join(f"- {w}" for w in response.warnings))

            logger.info(
                "Chat response emitted",
                extra={
                    "sources": len(getattr(response, "sources", []) or []),
                    "warnings": len(getattr(response, "warnings", []) or []),
                },
            )

    finally:
        if tmp_root is not None:
            tmp_root.cleanup()
        if tmp_base is not None:
            tmp_base.cleanup()
