from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import typer
from langchain_openai import OpenAIEmbeddings
from rich.console import Console

from mana_analyzer.analysis.checks import PythonStaticAnalyzer
from mana_analyzer.analysis.chunker import CodeChunker
from mana_analyzer.config.settings import Settings, default_index_dir
from mana_analyzer.llm.ask_agent import AskAgent
from mana_analyzer.llm.analyze_chain import AnalyzeChain
from mana_analyzer.llm.qna_chain import QnAChain
from mana_analyzer.llm.repo_chain import RepositoryMultiChain
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

from mana_analyzer.services.vulnerability_service import VulnerabilityService
from mana_analyzer.services.report_service import ReportService

app = typer.Typer(help="mana-analyzer CLI")
console = Console()
logger = logging.getLogger(__name__)
OUTPUT_DIR: Path | None = None


def _index_has_vectors(index_dir: Path) -> bool:
    faiss_dir = index_dir / "faiss"
    if not faiss_dir.exists() or not faiss_dir.is_dir():
        return False
    return any(faiss_dir.iterdir())


def _index_has_chunks(index_dir: Path) -> bool:
    chunks_file = index_dir / "chunks.jsonl"
    return chunks_file.exists() and chunks_file.stat().st_size > 0


def _index_has_search_data(index_dir: Path) -> bool:
    return _index_has_vectors(index_dir) or _index_has_chunks(index_dir)


# ----------------------------
# Report artifact helpers
# ----------------------------

def _resolve_report_artifact_dir(project_root: Path) -> Path:
    # Default artifact directory if --output-dir not provided:
    # <project_root>/.mana_logs
    if OUTPUT_DIR is not None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return OUTPUT_DIR
    target = project_root / ".mana_logs"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _resolve_report_artifact_paths(target_path: str | Path) -> tuple[Path, Path]:
    target = Path(target_path).resolve()
    project_root = target if target.is_dir() else target.parent
    stamp = datetime.now().strftime("%Y%m%d-%H")
    stem = f"{project_root.name}-{stamp}-report"
    out_dir = _resolve_report_artifact_dir(project_root)
    return out_dir / f"{stem}.json", out_dir / f"{stem}.md"


def _resolve_out_path(arg: str | None, default_path: Path, *, suffix: str) -> Path:
    """
    Accept either:
      - None => default_path
      - directory path => <dir>/<default_name>
      - file path => file path
    """
    if not arg:
        return default_path
    p = Path(arg).expanduser().resolve()
    if p.exists() and p.is_dir():
        return p / default_path.name
    # if user passed "logs/" but directory doesn't exist yet, treat ending "/" as directory intent
    if str(arg).endswith(("/", "\\")):
        p.mkdir(parents=True, exist_ok=True)
        return p / default_path.name
    # if no suffix provided, ensure correct suffix
    if p.suffix == "":
        return p.with_suffix(suffix)
    return p


def _clamp_detail_line_target(value: int) -> int:
    if value < 300:
        return 300
    if value > 400:
        return 400
    return value


def build_store(settings: Settings) -> FaissStore:
    logger.debug(
        "Building vector store embeddings client",
        extra={
            "embed_model": settings.openai_embed_model,
            "has_base_url": bool(settings.openai_base_url),
        },
    )
    kwargs = {"api_key": settings.openai_api_key, "model": settings.openai_embed_model}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    embeddings = OpenAIEmbeddings(**kwargs)
    return FaissStore(embeddings=embeddings)


def build_index_service(settings: Settings) -> IndexService:
    logger.debug("Initializing index service")
    return IndexService(parser=MultiLanguageParser(), chunker=CodeChunker(), store=build_store(settings))


def build_search_service(settings: Settings) -> SearchService:
    logger.debug("Initializing search service")
    return SearchService(store=build_store(settings))


def build_analyze_service() -> AnalyzeService:
    logger.debug("Initializing analyze service")
    return AnalyzeService(analyzer=PythonStaticAnalyzer())


def build_llm_analyze_service(settings: Settings, model_override: str | None) -> LlmAnalyzeService:
    model = model_override or settings.openai_chat_model
    logger.debug("Initializing LLM analyze service", extra={"chat_model": model})
    chain = AnalyzeChain(
        api_key=settings.openai_api_key,
        model=model,
        base_url=settings.openai_base_url,
    )
    return LlmAnalyzeService(analyze_chain=chain)


def build_ask_service(settings: Settings, model_override: str | None) -> AskService:
    model = model_override or settings.openai_chat_model
    logger.debug("Initializing ask service", extra={"chat_model": model})
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
        project_root=Path.cwd(),
    )
    return AskService(store=build_store(settings), qna_chain=qna, ask_agent=agent, search_service=search_service)


def build_dependency_service() -> DependencyService:
    logger.debug("Initializing dependency service")
    return DependencyService()


def build_repo_chain(settings: Settings, model_override: str | None) -> RepositoryMultiChain:
    model = model_override or settings.openai_chat_model
    logger.debug("Initializing repository multi-chain", extra={"chat_model": model})
    return RepositoryMultiChain(
        api_key=settings.openai_api_key,
        model=model,
        base_url=settings.openai_base_url,
    )


def build_describe_service(settings: Settings, model_override: str | None, use_llm: bool) -> DescribeService:
    llm_chain = build_repo_chain(settings, model_override) if use_llm else None
    return DescribeService(dependency_service=build_dependency_service(), llm_chain=llm_chain)


def build_report_service(
    *,
    use_llm: bool,
    model_override: str | None,
    include_tests: bool,
) -> ReportService:
    # Only load Settings if we need LLM
    settings = Settings() if use_llm else None

    dependency_service = build_dependency_service()
    analyze_service = build_analyze_service()

    llm_analyze_service = None
    if use_llm:
        assert settings is not None
        llm_analyze_service = build_llm_analyze_service(settings, model_override=model_override)
        describe_service = build_describe_service(settings, model_override=model_override, use_llm=True)
    else:
        # Avoid Settings() so OpenAI creds are not required
        describe_service = DescribeService(dependency_service=dependency_service, llm_chain=None)

    structure_service = StructureService(include_tests=include_tests)
    vuln_service = VulnerabilityService()

    return ReportService(
        dependency_service=dependency_service,
        analyze_service=analyze_service,
        llm_analyze_service=llm_analyze_service,
        describe_service=describe_service,
        structure_service=structure_service,
        vulnerability_service=vuln_service,
    )


def _resolve_output_file(target_path: str | Path | None = None) -> Path | None:
    if OUTPUT_DIR is None:
        return None
    target = Path(target_path).resolve() if target_path else Path.cwd().resolve()
    root = target if target.is_dir() else target.parent
    stamp = datetime.now().strftime("%Y%m%d-%H")
    file_name = f"{root.name}-{stamp}.log"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / file_name


def _append_output(output_file: Path | None, content: str) -> None:
    if output_file is None or not content:
        return
    with output_file.open("a", encoding="utf-8") as handle:
        handle.write(content.rstrip() + "\n")


def _emit_text(content: str, output_file: Path | None) -> None:
    console.print(content)
    _append_output(output_file, content)


def _emit_json(payload: object, output_file: Path | None) -> None:
    text = json.dumps(payload)
    console.print_json(text)
    _append_output(output_file, json.dumps(payload, indent=2))


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logs."),
    log_dir: str | None = typer.Option(None, "--log-dir", help="Directory for application log files."),
    output_dir: str | None = typer.Option(None, "--output-dir", help="Directory for saving command output logs."),
) -> None:
    global OUTPUT_DIR
    OUTPUT_DIR = Path(output_dir).resolve() if output_dir else None
    log_file = setup_logging(verbose=verbose, log_dir=log_dir)
    logger.debug("CLI initialized", extra={"verbose": verbose, "log_file": str(log_file)})


@app.command()
def index(
    path: str,
    index_dir: str | None = typer.Option(None, "--index-dir"),
    rebuild: bool = typer.Option(False, "--rebuild"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    output_file = _resolve_output_file(path)
    logger.info("Index command started", extra={"path": path, "rebuild": rebuild})
    settings = Settings()
    resolved_index_dir = Path(index_dir).resolve() if index_dir else default_index_dir(path)
    logger.debug("Resolved index directory", extra={"index_dir": str(resolved_index_dir)})
    service = build_index_service(settings)
    result = service.index(target_path=path, index_dir=resolved_index_dir, rebuild=rebuild)
    logger.info("Index command completed", extra=result)
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
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    output_file = _resolve_output_file()
    logger.info("Search command started", extra={"query": query, "k": k})
    settings = Settings()
    resolved_k = k or settings.default_top_k
    resolved_index_dir = Path(index_dir).resolve() if index_dir else default_index_dir(Path.cwd())
    logger.debug(
        "Resolved search parameters",
        extra={"k": resolved_k, "index_dir": str(resolved_index_dir)},
    )
    service = build_search_service(settings)
    results = service.search(index_dir=resolved_index_dir, query=query, k=resolved_k)
    logger.info("Search command completed", extra={"hits": len(results)})
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
    static_findings = service.analyze(path)
    llm_findings = []
    if with_llm:
        logger.info("LLM analyze mode enabled")
        llm_service = build_llm_analyze_service(settings, model_override=model)
        llm_findings = llm_service.analyze(path, static_findings=static_findings, max_files=llm_max_files)
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
        structure_service = StructureService(include_tests=include_tests)
        report = structure_service.analyze_project(path)
        describe_service = build_describe_service(settings, model_override=model, use_llm=with_llm)
        summary_report = describe_service.describe(
            path,
            max_files=llm_max_files,
            include_functions=False,
            use_llm=with_llm,
        )
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
        deps_report = build_dependency_service().analyze(path)
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
    logger.info("Ask command started", extra={"question": question, "k": k, "model_override": model})
    settings = Settings()
    resolved_k = k or settings.default_top_k
    resolved_index_dir = Path(index_dir).resolve() if index_dir else default_index_dir(Path.cwd())
    service = build_ask_service(settings, model_override=model)
    if dir_mode:
        root = Path(root_dir).resolve() if root_dir else Path.cwd().resolve()
        if root.is_file():
            root = root.parent
        logger.debug(
            "Resolved ask dir-mode parameters",
            extra={
                "k": resolved_k,
                "root_dir": str(root),
                "max_indexes": max_indexes,
                "auto_index_missing": auto_index_missing,
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
        if discovered_subprojects:
            for subproject in discovered_subprojects:
                expected_index = (subproject.root_path / ".mana_index").resolve()
                has_index_dir = expected_index in discovered_index_set
                has_search_data = has_index_dir and _index_has_search_data(expected_index)
                if has_search_data:
                    selected_indexes.append(expected_index)
                    continue
                if auto_index_missing:
                    try:
                        index_service.index(target_path=subproject.root_path, index_dir=expected_index, rebuild=False)
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
            root_index = (root / ".mana_index").resolve()
            if root_index.exists() and _index_has_search_data(root_index):
                selected_indexes = [root_index]
            elif auto_index_missing:
                try:
                    index_service.index(target_path=root, index_dir=root_index, rebuild=False)
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
        logger.debug(
            "Resolved ask parameters",
            extra={"k": resolved_k, "index_dir": str(resolved_index_dir)},
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
    settings = Settings()
    dependency_service = build_dependency_service()
    report = dependency_service.analyze(path)

    if use_llm:
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

    if rules_file:
        rules = Path(rules_file).read_text(encoding="utf-8", errors="ignore").lower()
        if "django" in rules:
            report.frameworks = sorted(set(report.frameworks) | {"Django"})
            report.technologies = sorted(set(report.technologies) | {"Django"})
        if "react" in rules:
            report.frameworks = sorted(set(report.frameworks) | {"React"})
            report.technologies = sorted(set(report.technologies) | {"React"})

    if output_json:
        Path(output_json).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    if output_dot:
        Path(output_dot).write_text(report.to_dot(), encoding="utf-8")
    if output_graphml:
        Path(output_graphml).write_text(report.to_graphml(), encoding="utf-8")

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
    report = build_dependency_service().analyze(path)
    if output_dot:
        Path(output_dot).write_text(report.to_dot(), encoding="utf-8")
    if output_graphml:
        Path(output_graphml).write_text(report.to_graphml(), encoding="utf-8")

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
    output_format: str = typer.Option("both", "--output-format"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    output_file = _resolve_output_file(path)
    settings = Settings()
    service = build_describe_service(settings, model_override=model, use_llm=use_llm)
    report = service.describe(path, max_files=max_files, include_functions=functions, use_llm=use_llm)
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
    # NEW deep profile options:
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
        # deep implies structure analysis (rendering needs it)
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

    payload = report_obj.to_dict()
    markdown = service.render_markdown(report_obj)

    # Write artifacts
    if output_format in {"json", "both"}:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if output_format in {"markdown", "both"}:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(markdown, encoding="utf-8")

    # Console behavior
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