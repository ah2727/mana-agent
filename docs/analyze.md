# Analyze

This project provides a unified **`analyze`** command that runs a full repository pass and produces consolidated reports (static findings, LLM findings, description, structure analysis, and dependency vulnerability scanning).

> Primary implementation: `src/mana_agent/commands/analyze_cli.py`.

## What the command does (pipeline)

The `analyze` CLI (Typer) performs the following high-level steps:

1. Resolve project root and load settings.
2. Build dependency/inventory report.
3. Build/update a vector index for search.
4. Optionally run semantic search by `--query`.
5. Run static analysis (Python).
6. Optionally run LLM-assisted analysis across a subset of files.
7. Merge and de-duplicate findings.
8. Generate repository description + per-file summaries.
9. Generate structure analysis.
10. Scan dependencies for known vulnerabilities (OSV).
11. Render outputs:
   - **Markdown** unified report
   - **HTML** report
   - **JSON** payload (optional)
   - plus dependency graph outputs (`.mana/analyze.dot` and `.mana/analyze.graphml`)

See `src/mana_agent/commands/analyze_cli.py` for the exact ordering.

## CLI usage

```bash
mana-agent analyze PATH [--query TEXT] [options]
```

Common options:

- `--query TEXT` / `--k INT` : run search against the indexed repository.
- `--model MODEL` : override LLM model.
- `--llm-max-files INT` : limit number of files to send to the LLM.
- `--summary-max-files INT` : limit number of files summarized in the description stage.
- `--include-tests / --no-include-tests` : include test files in parsing/description.
- `--online / --offline` : enable/disable online vulnerability scanning.
- `--security-scope {all,runtime,dev}` : choose which dependency scopes to scan.
- `--security-lens {defensive-red-team,architecture,compliance}` : affects security reporting flavor.
- `--report-profile {standard,deep}` : controls report depth.
- `--output-format {json,markdown,html,all}` : choose generated artifacts.
- `--fail-on {none,warning,error}` : process exit code policy.
- `--json` : emit JSON to stdout in addition to artifacts.

### Example

```bash
mana-agent analyze . \
  --query "authentication" \
  --output-format all \
  --fail-on warning
```

## Outputs

Given project root `PATH`, the command writes artifacts under that root:

- `PATH/<...>/analyze.json` (only if `--output-format json|all`)
- `PATH/<...>/analyze.md` (only if `--output-format markdown|all`)
- `PATH/<...>/analyze.html` (only if `--output-format html|all`)
- `PATH/.mana/analyze.dot` (dependency graph)
- `PATH/.mana/analyze.graphml` (dependency graph)

Additionally, `.mana` artifacts may be used/consumed by other tooling.

## Finding model

The command merges two sources of findings:

- **Static**: produced by `PythonStaticAnalyzer` / `AnalyzeService`.
- **LLM**: produced by a dedicated LLM analyze service (if enabled).

Merged results are de-duplicated by `(rule_id, severity, file_path, line, column, message)`.

See `_merge_findings(...)` and the static analysis service in `src/mana_agent/services/analyze_service.py`.

## Static analysis implementation details

Static analysis is implemented in `src/mana_agent/services/analyze_service.py`.

Notable behavior:

- Collects files using `iter_python_files(target)`.
- Uses a `ProcessPoolExecutor` when multiple workers are available.
- Sorts findings by `(file_path, line, column, rule_id)` before returning.

## Security scanning

Vulnerability scanning is performed by `VulnerabilityService().scan_dependencies(...)`.

The CLI passes:

- `online` flag
- OSV timeout
- dependency scope (`--security-scope`)

Any warnings from the vulnerability scan are appended to the overall report warnings.

## Fail behavior

- `--fail-on warning`: exits with code `1` if any finding severity is `warning` or `error`.
- `--fail-on error`: exits with code `1` if any finding severity is `error`.

Otherwise, it completes successfully.
