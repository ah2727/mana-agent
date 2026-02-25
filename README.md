
**Installable Python CLI AI code analyzer for multi‑language repositories**  

---  

## 📖 Overview  

`mana‑analyzer` is a command‑line tool that blends static code analysis with LLM‑powered retrieval‑augmented generation (RAG).  
It can index a repository (including Python, JavaScript/TypeScript, Dart, JVM languages, Go, Rust, PHP, Ruby, C/C++ and common markup), run fast static checks,
and answer natural‑language questions about the code‑base while providing precise file‑line citations.

Key capabilities:  

| Capability | Description |
|-----------|-------------|
| **Incremental indexing** | Stores embeddings in a local FAISS vector store; only changed files are re‑indexed. |
| **Static checks** | Detects unused/wildcard imports, missing docstrings, deep nesting, etc. (parallelised). |
| **Semantic search** | Multi‑index, multi‑threaded search over the whole code‑base. |
| **RAG‑powered `ask`** | Retrieves relevant chunks, feeds them to an LLM, and returns answers with citations. |
| **Dependency & tech detection** | Parses Python & JS/TS manifests, builds a dependency graph, and can export to JSON/DOT/GraphML. |
| **`describe` pipeline** | Selects the most “interesting” files, summarises them, and synthesises an architecture overview. |
| **Security scan** | Runs `pip list --outdated` + `safety check --full-report`. |
| **Chat mode** | Interactive REPL that re‑uses the `ask` service, optionally with tool‑aware reasoning. |
| **Extensible logging** | Separate logs for the application and for every LLM request/response. |

The core README currently ships with the project (`README.md` lines 1‑108) and already lists most of these 
features【/Users/ah/Documents/mana-agent/README.md:1-108】.

---  

## 🛠️ Requirements  

* Python **3.10+** (tested up to 3.14)  
* OpenAI (or compatible) API key – set `OPENAI_API_KEY` in `.env` or the environment.  

Optional: `OPENAI_BASE_URL` for non‑OpenAI providers (e.g. NVIDIA NIM)【/Users/ah/Documents/mana-agent/README.md:30-35】.

---  

## 🚀 Installation  

```bash
python3.10 -m venv .venv          # create an isolated env
source .venv/bin/activate          # activate it
pip install -r requirements.txt    # runtime deps
pip install -e .                   # install the package in editable mode
cp .env.example .env               # copy template env file
```

Set the API key: `export OPENAI_API_KEY=sk-…` (or edit `.env`).  

---  

## 📦 CLI Usage  

Below is a **canonical cheat‑sheet** (all commands accept `--json` for machine‑readable output).

```bash
# Index a repo (creates .mana_index at the root of the target)
mana-analyzer index /path/to/codebase

# Index with custom output dir and explicit index location
mana-analyzer --output-dir ./logs index /path/to/codebase --index-dir /tmp/my_index

# Semantic search
mana-analyzer search "function that validates token" \
    --index-dir /path/to/codebase/.mana_index

# Full static analysis (fails on warnings if asked)
mana-analyzer analyze /path/to/codebase --fail-on warning

# LLM‑augmented analysis (max 10 files by default)
mana-analyzer analyze /path/to/codebase \
    --with-llm --llm-max-files 10 --model gpt-4.1-mini --full-structure

# Natural‑language question answering
mana-analyzer ask "How does auth flow work?" \
    --index-dir /path/to/codebase/.mana_index

# Directory‑mode ask (multi‑project repositories)
mana-analyzer ask "Where are auth rules defined?" \
    --dir-mode --root-dir /path/to/mono-repo

# Ask with tool‑aware reasoning (agent)
mana-analyzer ask "How does billing work?" \
    --dir-mode --root-dir /path/to/mono-repo --agent-tools

# Dependency graph export
mana-analyzer deps /path/to/codebase --json \
    --dot /tmp/deps.dot --graphml /tmp/deps.graphml

# Visual graph generation only
mana-analyzer graph /path/to/codebase \
    --dot /tmp/deps.dot --graphml /tmp/deps.graphml

# Repository description (LLM summarisation)
mana-analyzer describe /path/to/codebase \
    --llm-model gpt-4.1-mini --max-files 20

# Security scan (pip outdated + safety)
mana-analyzer scan --requirements-file requirements.txt
```

All global options (`--verbose`, `--log-dir <path>`, `--output-dir <path>`) apply to every command【/Users/ah/Documents/mana-agent/README.md:84-113】.

---  

## 💬 Chat Mode (interactive REPL)  

Chat mode gives a conversational interface on top of the `ask` service.  

```bash
# Simple REPL (single index)
mana-analyzer chat --index-dir /path/to/.mana_index

# Multi‑project (directory) mode
mana-analyzer chat \
    --dir-mode \
    --root-dir /path/to/repo \
    --max-indexes 5 \
    --auto-index-missing   # create missing sub‑indexes on‑the‑fly
```

### How it works  

* `ChatService` wraps `AskService` and keeps an in‑memory history of `(question, answer)` pairs.  
* For each input it decides whether to call `AskService.ask()` (single index) or `AskService.ask_dir_mode()` (directory).  
* If `--agent-tools` is enabled the underlying *agent* may invoke additional helper tools, enabling deeper, multi‑step reasoning.  

See the “Chat Mode – mana‑analyzer” section of the existing README for a full description (lines 115‑260)【/Users/ah/Documents/mana-agent/README.md:115-260】.

---  

## 🏗️ Architecture Highlights  

* **Indexing pipeline** – walks the filesystem, extracts language‑specific tokens, creates embeddings, and writes them to a FAISS DB (`src/indexer.py`).  
* **Static analysis** – runs a pool of workers that apply AST‑based checks (`src/static_checks/…`).  
* **Dependency graph** – builds a directed graph from import statements and manifest files (`src/mana_analyzer/services/describe_service.py` contains the 
centrality heuristics for file selection【/Users/ah/Documents/mana-agent/src/mana_analyzer/services/describe_service.py:22-33】).  
* **RAG `ask`** – retrieves top‑k chunks, formats a prompt, calls the LLM, and returns `AskResponse` with citations.  
* **Logging** – application logs go to `<project_root>/.mana_logs/YYYY‑MM‑DD‑<root>.log`; LLM calls are recorded as JSONL in 
`<project_root>/.mana_llm_logs/...` (or overridden via `MANA_LLM_LOG_FILE`)【/Users/ah/Documents/mana-agent/README.md:121-150】.  

---  

## 🧪 Development  

```bash
# Install dev dependencies
pip install -e .

# Run the test suite
pytest
```

The project follows the standard `setuptools` layout (`src/` is the package root) as declared in 
`pyproject.toml`【/Users/ah/Documents/mana-agent/pyproject.toml:1-30】.

---  

## 🤝 Contributing  

1. Fork the repo.  
2. Create a feature branch (`git checkout -b my‑feature`).  
3. Keep the code lint‑clean and add tests.  
4. Submit a PR; CI will run the test suite and type‑checks.

---  

## 📄 License  

Distributed under the MIT License. See `LICENSE` for details.  

---  

## 📚 Full README (generated)  

Below is a **complete, polished README** that you can replace the existing `README.md` with. It incorporates the original sections, expands on the 
architecture, and adds a “Chat Mode” overview, development instructions, and contribution guidelines.

```markdown
# mana-analyzer

Installable Python CLI AI code analyzer for multi‑language repositories.

## Features
- Incremental indexing into local FAISS vector stores across Python, JS/TS, Dart, JVM, native, and common scripting/markup files.
- Static checks: unused imports, wildcard imports, missing docstrings, deep nesting (parallelized via process pools).
- Semantic search over indexed code with multi‑index threading for faster fan‑out.
- RAG‑powered `ask` command with file+line citations and concurrency‑aware upstream helpers.
- Dependency and technology detection across Python and JS/TS manifests/imports plus an async Safety‑backed scan.
- Multi‑step repository `describe` pipeline (file selection → summarisation → architecture synthesis).
- Dependency graph export to JSON, DOT, and GraphML.
- Interactive **Chat Mode** (REPL) with optional tool‑aware reasoning.
- Extensive logging (application logs + LLM request/response JSONL).

## Requirements
- Python 3.10+
- OpenAI API key (`OPENAI_API_KEY`)

Optional: `OPENAI_BASE_URL` for compatible providers (e.g., NVIDIA NIM).

## Setup
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```
Set `OPENAI_API_KEY` in `.env`.

## CLI Usage
```bash
# Index a repository
mana-analyzer index /path/to/codebase

# Search
mana-analyzer search "function that validates token" --index-dir /path/to/.mana_index

# Static analysis
mana-analyzer analyze /path/to/codebase --fail-on warning

# LLM‑augmented analysis
mana-analyzer analyze /path/to/codebase --with-llm --llm-max-files 10 --model gpt-4.1-mini --full-structure

# Ask a question
mana-analyzer ask "How does auth flow work?" --index-dir /path/to/.mana_index

# Directory mode ask (multiple sub‑projects)
mana-analyzer ask "Where are auth rules defined?" --dir-mode --root-dir /path/to/mono-repo

# Ask with agent tools
mana-analyzer ask "How does billing work?" --dir-mode --root-dir /path/to/mono-repo --agent-tools

# Dependency graph
mana-analyzer deps /path/to/codebase --json --dot /tmp/deps.dot --graphml /tmp/deps.graphml

# Visualise graph only
mana-analyzer graph /path/to/codebase --dot /tmp/deps.dot --graphml /tmp/deps.graphml

# Repository description
mana-analyzer describe /path/to/codebase --llm-model gpt-4.1-mini --max-files 20

# Security scan
mana-analyzer scan --requirements-file requirements.txt
```
All commands support `--json`, `--verbose`, `--log-dir <path>`, and `--output-dir <path>`.

## Chat Mode – Interactive REPL
```bash
# Single‑index chat
mana-analyzer chat --index-dir /path/to/.mana_index

# Multi‑project directory chat
mana-analyzer chat \
    --dir-mode \
    --root-dir /path/to/repo \
    --max-indexes 5 \
    --auto-index-missing \
    --agent-tools
```
Chat mode keeps a session‑local history, re‑uses the `AskService`, and can invoke tool‑aware agents when `--agent-tools` is set. Type `exit`, `quit`, or press 
`Ctrl+C` to leave.

## Architecture Overview
- **Indexer** (`src/indexer.py`): walks files, extracts language‑specific tokens, builds FAISS embeddings.
- **Static Checks** (`src/static_checks/`): parallel AST scans for imports, docstrings, nesting, etc.
- **Dependency Graph** (`src/mana_analyzer/services/describe_service.py`): builds a module edge graph and computes centrality to prioritize files for 
summarisation.
- **Ask Service** (`src/mana_analyzer/services/ask_service.py`): retrieves relevant chunks, constructs prompts, calls the LLM, and formats `AskResponse` with 
citations.
- **Chat Service** (`src/mana_analyzer/services/chat_service.py`): REPL wrapper around `AskService`, maintains conversation history, and delegates to 
single‑index or directory‑mode pipelines.
- **Logging**: app logs → `<repo>/.mana_logs/YYYY‑MM‑DD-<repo>.log`; LLM logs → `<repo>/.mana_llm_logs/...` (override via `MANA_LLM_LOG_FILE`).

## Development
```bash
pip install -e .   # install with test & lint deps
pytest                  # run tests
```
The project follows the standard `setuptools` layout (`src/` as the package root) as declared in `pyproject.toml`.

## Contributing
1. Fork the repo.  
2. Create a feature branch (`git checkout -b my-feature`).  
3. Keep the code lint‑clean, add tests, and ensure they pass.  
4. Open a Pull Request; CI will validate the changes.

## License
MIT License – see the `LICENSE` file.
