# mana-analyzer

**Installable Python CLI AI code analyzer for multi‑language repositories**

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick‑Start Guide](#quick-start-guide)
5. [Command‑Line Interface (CLI) Reference](#cli-reference)
6. [Tooling & Integrations](#tooling--integrations)
7. [Coding Flows & Debugging](#coding-flows--debugging)
8. [Architecture Overview](#architecture-overview)
9. [Development & Testing](#development--testing)
10. [Contributing](#contributing)
11. [License](#license)
12. [Contact & Support](#contact--support)

---

## Overview

`mana-analyzer` is a **tool-aware, LLM-augmented code analysis suite** that can:
- Incrementally index source code of many languages (Python, JavaScript/TypeScript, Dart, JVM languages, C/C++, Bash, PowerShell, HTML/Markdown, etc.).
- Perform static quality checks (unused imports, missing doc-strings, deep nesting, cyclomatic complexity, security-related patterns, …) in parallel using a process pool.
- Store vector embeddings in a local FAISS index and provide **semantic search** over the whole repository.
- Answer natural-language questions with Retrieval-Augmented Generation (RAG) while citing exact source lines.
- Generate dependency graphs (JSON/DOT/GraphML) and run optional security scans.
- Offer an interactive REPL (`chat`) that can invoke *tool-aware agents* for multi-step reasoning, including a **coding agent** capable of generating patches.

The project is deliberately modular: the CLI, core services, LLM wrappers, parsers, tool workers, and vector-store back-ends live in separate packages under `src/mana_analyzer/`. This layered design makes it easy to replace the LLM provider, swap the vector store, add new language parsers, or extend the agent tooling without touching the CLI itself. The service layer exposes reusable classes such as the indexing/searching pipelines, the `AskAgent`, and `CodingAgent`, all wired together in `src/mana_analyzer/commands/cli.py`.

---

## Features

- **Incremental indexing** – only newly added or modified files are re-embedded, reducing compute cost.
- **Multi-language support** – parsers for Python, JS/TS, Dart, JVM, native C/C++, Bash, PowerShell, HTML/Markdown, etc.
- **Static analysis suite** – unused imports, wildcard imports, missing doc-strings, deep nesting, cyclomatic complexity, security patterns, and more.
- **Semantic search** – fast FAISS-backed similarity lookup with configurable distance metric.
- **RAG `ask` command** – retrieve relevant chunks, construct prompts, call LLM, and return answers with line citations.
- **Dependency graph generation** – directed graphs of import relationships; export formats: JSON, DOT, GraphML.
- **Security scanning** – integrates `safety` vulnerability database and `pip list --outdated`.
- **Chat REPL** – stateful conversation with the `AskService`; optional *agent-tools* mode enables tool-calling.
- **Coding agent** – a helper that can generate code, apply patches, and iteratively improve solutions.
- **Configurable logging** – per-run logs, LLM call logs, and optional JSONL for downstream analysis.
- **Extensible architecture** – plug-in new parsers, replace FAISS with another vector store, or switch LLM back-ends (OpenAI, Azure, local LM, etc.).
- **GitHub code search tool** – introduces a `github_code_search` LangChain tool that wraps the GitHub code search API. The tool provides normalized payloads with `ok`, `results`, and `error` fields, respects `GITHUB_TOKEN`, and gracefully reports rate limits and other failures.

---

## Installation

```bash
# 1️⃣ Create an isolated virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# 2️⃣ Upgrade pip and install the package in editable mode
pip install --upgrade pip
pip install -e .[dev]
```

The package declares the following optional extras:
- `dev` – testing, linting, and additional development tools.
- `faiss-gpu` – GPU-accelerated FAISS (requires CUDA).
- `security` – `safety` for vulnerability scanning.

Make sure you have an OpenAI-compatible API key available, e.g.:
```bash
export OPENAI_API_KEY="sk-…"
# Optional: point to a self-hosted endpoint
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

---

## Quick-Start Guide

```bash
# Index a repository (creates a FAISS index under ~/.cache/mana_analyzer)
mana-analyzer index /path/to/your/project

# Perform a semantic search
mana-analyzer search "how does pagination work"

# Ask a natural-language question (RAG)
mana-analyzer ask "What are the security risks in the authentication module?"

# Start an interactive chat session
mana-analyzer chat
```

All commands accept a `--verbose` flag for more detailed output and a `--config` flag to point to a custom `settings.toml`.

---

## Command-Line Interface (CLI) Reference

| Command | Description |
|---------|-------------|
| `index <path>` | Walks the given directory, parses files, creates embedding chunks, and stores them in a FAISS index. |
| `search <query>` | Performs semantic similarity search against the local index and prints the top matches with file/line context. |
| `ask <question>` | Retrieval-augmented generation: fetches relevant chunks, builds a prompt, calls the LLM, and returns an answer with citations. |
| `chat` | Starts a REPL where you can ask multiple questions; the session retains context and can invoke tool-aware agents. |
| `profile` | Runs the indexing pipeline with `cProfile` and writes a performance report to `profile.txt`. |
| `lint` | Executes the static analysis checks and prints a summary of warnings/errors. |
| `dependency-graph` | Generates a dependency graph of the indexed project; use `--format dot|graphml|json`. |
| `security-scan` | Runs `safety` against the project's dependencies and reports known vulnerabilities. |

All commands share common options:
- `--index-dir <dir>` – location of the FAISS index (default: `~/.cache/mana_analyzer`).
- `--log-level <LEVEL>` – Python logging level (`DEBUG`, `INFO`, `WARNING`, …).
- `--max-workers <N>` – number of parallel workers for indexing (defaults to the number of CPU cores).

---

## Tooling & Integrations

### Tool-aware agents and workers

- The REPL session hosts *tool-aware agents* that call LangChain `StructuredTool`s for internet search, repository search, or GitHub search. Each tool is registered through `src/mana_analyzer/tools/__init__.py`, so new tooling becomes available to every agent automatically.
- Background workers (`src/mana_analyzer/llm/tool_worker_process.py`) manage LLM calls and tool executions. They expose a lightweight payload protocol defined by the `WorkerInitPayload`, `ToolRunRequest`, and `ToolRunResponse` dataclasses, which also power the coding agent's patch submission workflow.

### GitHub code search service

- `safe_github_search` is a thin wrapper around the public GitHub code search API. It normalizes the response into the same schema as the `search_internet` tool so commands and agents can treat both uniformly.
- The tool honors `GITHUB_TOKEN` when provided and caps the returned results to five entries to stay within GitHub's rate limits. When a request fails (timeout, rate limit, network), it returns a friendly error payload instead of raising an exception, keeping long-running chats stable.
- Coding agents that need examples from external projects can now call `github_code_search` to retrieve relevant code snippets before generating answers or patches.

---

## Coding Flows & Debugging

When `chat` runs with `--coding-agent --coding-memory`, the coding agent persists flow state in
`<project>/.mana_index/chat_memory.sqlite3` so follow-up turns can reuse objective/context/checklists.

Use the new command to inspect flow state outside chat:

```bash
# Active flow summary
mana-analyzer flow .

# Explicit flow id as JSON
mana-analyzer flow . --flow-id <flow_id> --format json
```

Inside chat, `/flow` helpers are still available:

```text
/flow show
/flow checklist
/flow checkpoint
/flow reset
```

Detailed flow schema, planner/fallback lifecycle, and debugging guidance:
[`docs/coding-flows.md`](docs/coding-flows.md).

---

## Architecture Overview

```
src/
├─ mana_analyzer/
│  ├─ commands/          # Typer-based CLI entry points
│  ├─ analysis/          # Static analysis utilities
│  ├─ llm/               # LLM wrappers, prompts, and agents
│  ├─ parsers/           # Language-specific parsers → code chunks
│  ├─ services/          # Core business logic (index, search, ask, …)
│  ├─ utils/             # Helper functions (logging, discovery, etc.)
│  └─ vector_store/      # FAISS implementation
└─ tests/                # pytest suite
```

Key layers:
- **CLI layer** – `src/mana_analyzer/commands/cli.py` wires Typer commands to service classes.
- **Service layer** – Each feature (indexing, search, ask, chat, etc.) lives in its own service class, making them reusable outside the CLI.
- **LLM layer** – `AskAgent`, `CodingAgent`, and various LangChain Chains encapsulate prompt engineering and LLM interaction.
- **Vector store layer** – `FaissStore` abstracts FAISS index creation, upserts, and similarity queries.
- **Parser layer** – `MultiLanguageParser` delegates to language-specific parsers that produce `CodeChunk` objects.

---

## Development & Testing

```bash
# Install development dependencies
pip install -e .[dev]

# Run the test suite
pytest -q

# Lint and type-check
ruff check src tests
mypy src tests
```

The repository includes a smoke test (`tests/test_cli_smoke.py`) that exercises the most common CLI commands against a tiny synthetic project.

---

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feat/my-feature`).
3. Write tests for any new functionality.
4. Run the full test suite and ensure linting passes.
5. Open a Pull Request describing the change.


---

## License

`mana-analyzer` is released under the **MIT License**. See the `LICENSE` file for full details.

---

## Contact & Support

For bug reports or feature requests, please use the repository's issue tracker. For questions about usage, join the discussion channel (see CONTRIBUTING.md for preferred community links).

---
