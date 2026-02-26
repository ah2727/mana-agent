# mana-analyzer

**Installable Python CLI AI code analyzer for multi‑language repositories**

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Quick‑Start Guide](#quick-start-guide)
6. [Command‑Line Interface (CLI) Reference](#cli-reference)
7. [Architecture Deep‑Dive](#architecture)
   - [Package Layout](#package-layout)
   - [Core Modules](#core-modules)
   - [Service Layer](#service-layer)
   - [LLM Integration](#llm-integration)
   - [Vector Store (FAISS)](#vector-store)
   - [Utilities & Helpers](#utilities)
8. [Development & Testing](#development-testing)
9. [Contributing](#contributing)
10. [License](#license)
11 [Acknowledgements](#acknowledgements)

---

## Overview

`mana-analyzer` is a **tool‑aware, LLM‑augmented code analysis suite** that can:
- Incrementally index source code of many languages (Python, JavaScript/TypeScript, Dart, JVM languages, native C/C++, and common scripting/markup files).
- Perform static quality checks (unused imports, missing doc‑strings, deep nesting, etc.) in parallel via process pools.
- Enable semantic search over the indexed code using a local FAISS vector store.
- Provide Retrieval‑Augmented Generation (RAG) answering with **line‑level citations**.
- Detect dependencies, generate dependency graphs (JSON/DOT/GraphML), and surface security‑related alerts.
- Run an interactive REPL (`chat`) that can invoke *tool‑aware agents* for multi‑step reasoning.

The project is deliberately modular: the CLI, core services, LLM wrappers, parsers, and vector‑store back‑ends live in separate packages under `src/mana_analyzer/`.  This makes it easy to replace the LLM provider, swap the vector store, or add new language parsers.

---

## Features

- **Incremental indexing** – only newly added/modified files are re‑embedded, reducing compute cost.
- **Multi‑language support** – parsers for Python, JS/TS, Dart, JVM, native C/C++, Bash, PowerShell, HTML/Markdown, etc.
- **Static analysis checks** – unused imports, wildcard imports, missing doc‑strings, deep nesting, cyclomatic complexity, and more.
- **Semantic search** – fast FAISS‑backed similarity lookup with optional multi‑index fan‑out.
- **RAG `ask` command** – retrieve relevant chunks, construct prompts, call LLM, and return answers with citations.
- **Dependency graph generation** – creates directed graphs of import relationships; export formats: JSON, DOT, GraphML.
- **Security scanning** – combines `pip list --outdated` with `safety` vulnerability database.
- **Chat REPL** – stateful conversation with the `AskService`; optional *agent‑tools* mode enables tool‑calling.
- **Chat mode with Coding Agent** – the `chat` command accepts the `--coding-agent` flag to activate the optional `coding_agent` helper, which can generate code snippets, apply patches, and iteratively improve solutions.
- **Configurable logging** – per‑run logs, LLM call logs, and optional JSONL for downstream analysis.
- **Extensible architecture** – plug‑in new parsers, replace FAISS with a different vector store, or switch LLM back‑ends (OpenAI, Azure, local LM, etc.).

---

## Requirements

- **Python 3.10+** – recommended 3.12/3.13.  The code is type‑annotated and works with newer versions (3.14 may emit compatibility warnings).
- **OpenAI compatible API key** – set `OPENAI_API_KEY` in `.env` or the environment.  Alternatives (`OPENAI_BASE_URL`) are supported for hosted providers (e.g. NVIDIA NIM, Anthropic, Azure).
- **FAISS** – installed via `requirements.txt`; optional GPU build if available.
- **Optional dependencies** – `safety` for security scans, `graphviz` for DOT rendering, `networkx` for GraphML export.

---

## Installation

```bash
# 1️⃣ Create an isolated virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# 2️⃣ Install runtime dependencies
pip install -r requirements.txt

# 3️⃣ Install the editable package (allows live code changes)
pip install -e .

# 4️⃣ Copy the example environment file and set your API key
cp .env.example .env
# edit .env or export directly:
export OPENAI_API_KEY=sk-…
```

> **Tip** – For development, also install the test extras:
> ```bash
> pip install -e .[dev]
> ```

---

## Quick‑Start Guide

Below is a **canonical cheat‑sheet** that demonstrates the most common workflows.  All commands accept a global `--json` flag for machine‑readable output.

```bash
# 🔎 Index a repository (creates .mana_index at the repo root)
mana-analyzer index /path/to/codebase

# 📂 Index with a custom output directory and explicit index location
mana-analyzer --output-dir ./logs index /path/to/codebase \
    --index-dir /tmp/my_index

# 🪄 Semantic search across the indexed code
mana-analyzer search "function that validates token" \
    --index-dir /path/to/codebase/.mana_index

# 🛠️ Full static analysis (fails on warnings if requested)
mana-analyzer analyze /path/to/codebase --fail-on warning

# 🤖 LLM‑augmented analysis (process up to 10 files by default)
mana-analyzer analyze /path/to/codebase \
    --with-llm --llm-max-files 10 \
    --model gpt-4.1-mini --full-structure

# ❓ Natural‑language question answering (RAG)
mana-analyzer ask "How does auth flow work?" \
    --index-dir /path/to/codebase/.mana_index

# 📁 Directory‑mode ask (multi‑project repositories)
mana-analyzer ask "Where are auth rules defined?" \
    --dir-mode --root-dir /path/to/mono-repo

# 🧠 Ask with tool‑aware reasoning (agent) – can call helpers like `search` or `index` internally
mana-analyzer ask "How does billing work?" \
    --dir-mode --root-dir /path/to/mono-repo --agent-tools

# 💬 Chat REPL (interactive).  Add `--coding-agent` to enable code‑generation assistance.
mana-analyzer chat --coding-agent

# 📊 Dependency graph export (JSON/DOT/GraphML)
mana-analyzer deps /path/to/codebase --json \
    --dot /tmp/deps.dot --graphml /tmp/deps.graphml

# 🖼️ Visual graph generation only (no analysis)
mana-analyzer graph /path/to/codebase \
    --dot /tmp/deps.dot --graphml /tmp/deps.graphml

# 📖 Repository description (LLM summarisation)
mana-analyzer describe /path/to/codebase \
    --llm-model gpt-4.1-mini --max-files 20

# 🔐 Security scan (pip outdated + safety DB)
mana-analyzer scan --requirements-file requirements.txt
```

All global options (`--verbose`, `--log-dir <path>`, `--output-dir <path>`) apply to every command.

---

## CLI Reference

| Command | Description | Important Options |
|---------|-------------|-------------------|
| `index` | Walk a filesystem, parse source files, embed chunks, and store them in a FAISS DB. | `--index-dir`, `--output-dir`, `--exclude <glob>` |
| `search <query>` | Perform a vector similarity search against an existing index. | `--k`, `--score-threshold`, `--index-dir` |
| `analyze <path>` | Run static checks; optionally augment with LLM for deeper insights. | `--with-llm`, `--llm-max-files`, `--model`, `--full-structure` |
| `ask <question>` | RAG‑powered answer retrieval; returns citations. | `--dir-mode`, `--root-dir`, `--agent-tools`, `--json` |
| `chat` | Interactive REPL that wraps `ask`. | `--max-indexes`, `--auto-index-missing`, `--coding-agent` |
| `deps` | Build a dependency graph of imports/manifest entries. | `--json`, `--dot`, `--graphml` |
| `graph` | Export a visual representation of the dependency graph without analysis. | `--dot`, `--graphml` |
| `describe` | LLM summarisation of repository structure and purpose. | `--llm-model`, `--max-files` |
| `scan` | Security scan using `pip list --outdated` + `safety`. | `--requirements-file`, `--json` |

Each command lives in `src/mana_analyzer/commands/cli.py` and is wired through `argparse` (or `typer` in future versions).  See the source for additional flags.

---

## Architecture Deep‑Dive

### Package Layout

```
src/
└─ mana_analyzer/
   ├─ __init__.py                 # package entry point
   ├─ analysis/                   # AST‑based checks and chunking logic
   │   ├─ checks.py               # static rule implementations
   │   ├─ chunker.py              # turn source files into embedding‑ready chunks
   │   └─ models.py               # Pydantic models for check results
   ├─ commands/                   # CLI entry point (click/argparse wrapper)
   │   └─ cli.py                 # command dispatch
   ├─ config/                     # Settings handling via pydantic BaseSettings
   │   └─ settings.py            # env‑var backed config model
   ├─ llm/                        # LLM orchestration (LangChain wrappers)
   │   ├─ analyze_chain.py       # chain for static‑analysis augmentation
   │   ├─ ask_agent.py            # tool‑aware agent implementation
   │   ├─ coding_agent.py         # optional code‑generation helper
   │   ├─ prompts.py              # prompt templates (RAG, summarisation, etc.)
   │   └─ run_logger.py           # captures LLM call metadata
   ├─ parsers/                    # Language‑specific parsers
   │   ├─ python_parser.py       # Python AST extraction
   │   ├─ multi_parser.py        # dispatcher for multi‑language files
   │   └─ __init__.py
   └─ services/                   # High‑level business logic (service layer)
       ├─ analyze_service.py     # orchestration of static and LLM analysis
       ├─ ask_service.py          # RAG answer generation
       ├─ chat_service.py         # REPL state management
       ├─ dependency_service.py   # build import graph
       ├─ describe_service.py     # repository summarisation workflow
       ├─ index_service.py        # indexing orchestration
       ├─ llm_analyze_service.py  # LLM‑only analysis pipeline
       └─ report_service.py       # assembles final markdown / JSON reports
```
