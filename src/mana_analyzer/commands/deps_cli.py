from __future__ import annotations

from .cli_internal import *
from .output import build_output_sink

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
    sink = build_output_sink(command_name="deps", json_mode=as_json, output_file=output_file, console=console)
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
        sink.emit_json(report.to_dict())
    else:
        sink.emit_kv(
            title="Dependency Summary",
            items=[
                ("project", str(report.project_root)),
                ("languages", ", ".join(report.languages) if report.languages else "unknown"),
                ("frameworks", ", ".join(report.frameworks) if report.frameworks else "none"),
                ("package_managers", ", ".join(report.package_managers) if report.package_managers else "unknown"),
                ("runtime_dependencies", str(len(report.runtime_dependencies))),
                ("dev_dependencies", str(len(report.dev_dependencies))),
                ("module_edges", str(len(report.module_edges))),
                ("external_edges", str(len(report.dependency_edges))),
            ],
        )
