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


# CLI Chat Mode – mana-analyzer

This document describes the **Chat Mode** feature for the `mana-analyzer` CLI.  
Chat mode provides an interactive, conversational interface in the terminal, allowing users to explore repositories and ask follow-up questions without repeatedly invoking separate commands.

---

## ✨ Overview

The existing CLI exposes commands such as:

- `index`
- `search`
- `analyze`
- `ask`

However, these commands operate independently and require manual invocation each time.

**Chat Mode** introduces:

- 🖥 An interactive REPL (Read–Eval–Print Loop)
- 🧠 Context-aware multi-turn conversation
- 📂 Support for single-index and directory (multi-index) mode
- 🧰 Optional tool-aware answering
- 📄 JSON output support
- 🚪 Clean exit with `exit`, `quit`, or `Ctrl+C`

---

## 🏗 Architecture

Chat mode is built around a new service:


### ChatService Responsibilities

The `ChatService`:

- Wraps and internally builds an `AskService`
- Maintains in-memory conversation history
- Delegates questions to:
  - `AskService.ask()` (single index mode)
  - `AskService.ask_dir_mode()` (directory mode)
- Stores `(question, answer)` history
- Returns structured `AskResponse` objects

---

## 📦 Installation

After implementing the feature, install or reinstall the CLI:

```bash
pip install .

🚀 Usage
Basic Chat (Single Index Mode)
mana-analyzer chat --index-dir /path/to/.mana_index
Directory Mode (Multi-Project Repositories)
mana-analyzer chat \
  --dir-mode \
  --root-dir /path/to/repo \
  --max-indexes 5

Directory mode:

Automatically discovers sub-projects

Finds associated .mana_index directories

Optionally auto-creates missing indexes

⚙️ Command Options
Option	Description
--model	Override the default LLM
--index-dir	Path to a specific index directory
--k	Override top-k retrieval
--dir-mode	Enable multi-project directory mode
--root-dir	Root repository path for directory mode
--max-indexes	Limit discovered indexes (0 = unlimited)
--auto-index-missing / --no-auto-index-missing	Auto-create missing subproject indexes
--agent-tools	Enable tool-aware answering
--agent-max-steps	Maximum agent reasoning steps
--agent-timeout-seconds	Agent execution timeout
--json	Emit responses as JSON
💬 Interactive Session Example
mana-analyzer chat – type 'exit' or 'quit' to end.
💬 » What does the indexing pipeline do?

Output:

The indexing pipeline scans the repository, extracts metadata,
generates embeddings, and stores them in the vector index.

Sources:
- src/indexer.py:14-78
- src/embeddings.py:10-45

Exit with:

exit

Or press:

Ctrl+C
🧠 Conversation History

Chat mode stores conversation history in memory for the session:

self.history: list[tuple[str, str]]

This allows:

Future support for context-aware prompts

Extending the system to persist history to disk

Enhanced multi-turn reasoning

🛠 Internal Flow

User enters a question.

Input is normalized.

ChatService decides:

ask() → single index

ask_dir_mode() → directory mode

AskResponse returned.

Answer printed.

Sources + warnings displayed.

Turn added to history.

🧰 Agent Tool Mode

If --agent-tools is enabled:

Tool-aware reasoning becomes available

Specialized tools may be invoked

Supports deeper code understanding workflows

Example:

mana-analyzer chat --index-dir .mana_index --agent-tools
🔄 Async Support (Optional Enhancement)

Current implementation assumes synchronous calls.

If underlying services are async:

Wrap calls using asyncio.run()

Or make ChatService.ask() async

📤 JSON Output Mode

To emit structured output:

mana-analyzer chat --index-dir .mana_index --json

This prints serialized AskResponse objects.

📁 File Structure
mana_analyzer/
│
├── services/
│   ├── ask_service.py
│   ├── chat_service.py   ← New
│
├── cli.py                ← Updated with chat command
🎯 Goals Achieved

Chat Mode enables:

Iterative repository exploration

Faster developer workflows

Improved usability

Extensible architecture

Reuse of existing services

🔮 Future Enhancements

Persist chat history to disk

Context injection into prompts

Streaming token output

Session save/load

Multi-user support

Web-based UI layer

Rich terminal UI (e.g. rich or textual)

🏁 Summary

Chat Mode transforms mana-analyzer from a command-based CLI into a conversational development assistant inside the terminal.

It reuses the existing indexing and question-answering infrastructure while adding a clean interactive workflow.


mana-analyzer chat \
  --dir-mode \
  --root-dir /Users/ah/Documents/karlancer/loanbot \
  --agent-tools \
  --agent-max-steps 20 \
  --agent-timeout-seconds 120