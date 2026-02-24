# mana-analyzer

Installable Python CLI AI code analyzer for multi-language repositories.

## Features
- Incremental indexing into local FAISS vector stores across Python, JS/TS, Dart, JVM, native, and common scripting/markup files.
- Static checks: unused imports, wildcard imports, missing docstrings, deep nesting (parallelized via process pools).
- Semantic search over indexed code with multi-index threading for faster fan-out.
- RAG-powered `ask` command with file+line citations and concurrency-aware upstream helpers.
- Dependency and technology detection across Python and JS/TS manifests/imports plus an async Safety-backed scan.
- Multi-step repository `describe` pipeline (file selection -> summarization -> architecture synthesis).
- Dependency graph export to JSON, DOT, and GraphML.

## Requirements
- Python 3.10+
- OpenAI API key

## Setup
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`.

Optional: set `OPENAI_BASE_URL` for OpenAI-compatible providers (for example NVIDIA NIM).
```bash
OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
```

## CLI Usage
```bash
mana-analyzer index /path/to/codebase
mana-analyzer --output-dir ./dir index /path/to/codebase
mana-analyzer search "function that validates token" --index-dir /path/to/codebase/.mana_index
mana-analyzer analyze /path/to/codebase --fail-on warning
mana-analyzer analyze /path/to/codebase --with-llm --llm-max-files 10 --model gpt-4.1-mini --full-structure
mana-analyzer ask "How does auth flow work?" --index-dir /path/to/codebase/.mana_index
mana-analyzer ask "Where are auth rules defined?" --dir-mode --root-dir /path/to/mono-repo
mana-analyzer ask "How does billing work?" --dir-mode --root-dir /path/to/mono-repo --agent-tools
mana-analyzer --verbose --log-dir /tmp/mana-logs analyze /path/to/codebase
mana-analyzer deps /path/to/codebase --json
mana-analyzer graph /path/to/codebase --dot /tmp/deps.dot --graphml /tmp/deps.graphml
mana-analyzer describe /path/to/codebase --llm-model gpt-4.1-mini --max-files 20
mana-analyzer scan --requirements-file requirements.txt
```

Use global `--output-dir` to save command output logs to a file named:
`<analyze_root_name>-YYYYMMDD-HH.log` inside the chosen directory.

JSON output is available with `--json` on all commands.

`analyze --with-llm` is opt-in and will increase latency and token cost. By default it analyzes at most 10 files, prioritized by files with the most static findings (fallback: sorted files when there are no static findings). LLM findings use the same `Finding` schema as static findings.
The `scan` command runs `pip list --outdated` alongside `safety check --full-report`, writes an optional JSON report, and can fail fast if Safety reports vulnerabilities.

## Logging
- Application logs are written to files (not emitted by Python logging to console).
- Global options:
  - `--verbose` enables `DEBUG` logs.
  - `--log-dir <path>` writes app logs to the given directory.
- Default app log path (when `--log-dir` is not provided):
  - `<project_root>/.mana_logs/YYYY-MM-DD-<project_root_name>.log`
- LLM run logs are stored as JSONL and include full request/response payloads for each LLM run.
- Default LLM run log path:
  - `<project_root>/.mana_llm_logs/YYYY-MM-DD-<project_root_name>-runs.jsonl`
- Override LLM run log path with:
  - `MANA_LLM_LOG_FILE=/absolute/path/to/runs.jsonl`

## Command Contracts
- Global options: `--verbose`, `--log-dir <path>`, `--output-dir <path>`
- `mana-analyzer index <path> [--index-dir <path>] [--rebuild] [--json]`
- `mana-analyzer search <query> [--k <int>] [--index-dir <path>] [--json]`
- `mana-analyzer analyze <path> [--fail-on warning|error|none] [--with-llm] [--model <name>] [--llm-max-files <int>] [--tech-summary] [--chain-profile <name>] [--chain-config <json>] [--json]`
- `mana-analyzer ask "<question>" [--k <int>] [--model <name>] [--index-dir <path>] [--dir-mode] [--root-dir <path>] [--max-indexes <int>] [--auto-index-missing/--no-auto-index-missing] [--agent-tools] [--agent-max-steps <int>] [--agent-timeout-seconds <int>] [--json]`
- `mana-analyzer deps <path> [--llm] [--llm-model <name>] [--rules <path>] [--json-out <path>] [--dot <path>] [--graphml <path>] [--json]`
- `mana-analyzer graph <path> [--dot <path>] [--graphml <path>] [--json]`
- `mana-analyzer describe <path> [--llm/--no-llm] [--llm-model <name>] [--max-files <int>] [--functions] [--output-format json|markdown|both] [--json]`



## security analyze 
mana-analyzer --verbose --log-dir logs/ --output-dir logs/ report /Users/ah/Documents/karlancer/loanbot \                                                    1 ↵
  --with-llm \
  --output-format markdown \
  --report-profile deep \
  --detail-line-target 350 \
  --security-lens defensive-red-team \
  --json-out logs/ \
  --markdown-out logs/
