from __future__ import annotations

from .cli_internal import *
from .output import build_output_sink

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
    sink = build_output_sink(command_name="search", json_mode=as_json, output_file=output_file, console=console)
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
        sink.emit_json([item.to_dict() for item in results])
        return

    if not results:
        sink.emit_warning("No results found.")
        return

    rows: list[list[str]] = []
    for idx, hit in enumerate(results, start=1):
        rows.append(
            [
                str(idx),
                hit.symbol_name,
                f"{hit.score:.3f}",
                f"{hit.file_path}:{hit.start_line}-{hit.end_line}",
                hit.snippet[:180].strip(),
            ]
        )

    sink.emit_table(
        title=f"Search Results ({len(results)})",
        columns=["#", "symbol", "score", "location", "snippet"],
        rows=rows,
    )
