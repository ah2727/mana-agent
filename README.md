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
9. [Extending & Adding Language Support](#extending)
10. [Advanced Configuration & Performance Tuning](#advanced-config)
11. [FAQ](#faq)
12. [Contributing](#contributing)
13. [License](#license)
14. [Acknowledgements](#acknowledgements)
15. [Contact & Support](#contact)

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
# 🔍 Index a repository (creates .mana_index at the repo root)
mana-analyzer index /path/to/codebase

# 📂 Index with a custom output directory and explicit index location
mana-analyzer --output-dir ./logs index /path/to/codebase \
    --index-dir /tmp/my_index

# 🧠 Semantic search across the indexed code
mana-analyzer search "function that validates token" \
    --index-dir /path/to/codebase/.mana_index
# ⚙️ Full static analysis (fails on warnings if requested)
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

# 🛠️ Ask with tool‑aware reasoning (agent) – can call helpers like `search` or `index` internally
mana-analyzer ask "How does billing work?" \
    --dir-mode --root-dir /path/to/mono-repo --agent-tools

# 💬 Chat REPL (interactive).  Add `--coding-agent` to enable code‑generation assistance.
mana-analyzer chat --coding-agent
mana-analyzer chat --planning-mode
mana-analyzer chat --planning-max-questions <int> (new, default 3).

# 📈 Dependency graph export (JSON/DOT/GraphML)
mana-analyzer deps /path/to/codebase --json \
    --dot /tmp/deps.dot --graphml /tmp/deps.graphml

# 🖼️ Visual graph generation only (no analysis)
mana-analyzer graph /path/to/codebase \
    --dot /tmp/deps.dot --graphml /tmp/deps.graphml

# 📄 Repository description (LLM summarisation)
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

Each command lives in `src/mana_analyzer/commands/cli.py` and is wired through `typer` (future‑proofed for richer help output).  See the source for additional flags.

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
   ├─ commands/                  # CLI entry point (typer wrapper)
   │   └─ cli.py                  # command dispatch
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
   └─ services/                  # High‑level business logic (service layer)
       ├─ analyze_service.py     # orchestration of static and LLM analysis
       ├─ ask_service.py          # RAG answer generation
       ├─ chat_service.py         # REPL state management
       ├─ dependency_service.py   # build import graph
       ├─ describe_service.py     # repository summarisation workflow
       ├─ index_service.py        # indexing orchestration
       ├─ llm_analyze_service.py # LLM‑only analysis pipeline
       └─ report_service.py       # assembles final markdown / JSON reports
```

### Core Modules

- **`analysis.chunker`** – Implements `Chunk` dataclass with metadata (file path, start/end lines, language, token count).  Uses language‑specific tokenizers (tiktoken, tree‑sitter) to enforce a ~400‑token limit per chunk.
- **`llm.run_logger`** – Persists each LLM call (prompt, response, token usage, latency) to `logs/llm_calls.jsonl`.  Enables offline cost analysis and reproducibility.
- **`services.index_service`** – Coordinates file discovery, incremental hashing, chunk creation, and FAISS up‑sert.  Handles concurrency via `ProcessPoolExecutor`.

### Service Layer

All user‑facing commands delegate to a **service** object that is deliberately stateless (except for injected configuration).  This design enables:
1. Easy unit‑testing with mocks.
2. Re‑use of the same logic in a GUI or notebook context.
3. Future migration to an async‑first API without breaking the CLI.

### LLM Integration

The LLM stack is built on **LangChain** primitives but wrapped in thin adapters to keep the dependency surface small.  Prompt templates live in `llm/prompts.py` and are rendered with Jinja2 for clarity.  The default provider is **OpenAI**; to switch, set `OPENAI_BASE_URL` and `OPENAI_API_KEY` accordingly, or provide a custom `LLMClient` implementation.

### Vector Store (FAISS)

- Indexes are stored under `<repo_root>/.mana_index/` by default.
- Each language has its own flat index to avoid cross‑language token‑distribution bias.
- The `faiss.IndexFlatIP` (inner product) is used with normalized embeddings for cosine similarity.
- Periodic checkpointing writes `.meta` files containing the file‑hash map for incremental updates.

### Utilities & Helpers

- **`utils.path`** – safe pathlib helpers, glob‑based file iteration, and ignore‑pattern handling.
- **`utils.logging`** – structured console + file logging using `rich` for colourised output.
- **`utils.diff`** – simple unified‑diff generation used by the `coding_agent` when proposing patches.

---

## Development & Testing

The repository follows the **pytest** standard.  Tests reside in `tests/` and cover:
- Chunking correctness and token‑limit guarantees.
- Static analysis rule accuracy on synthetic code fixtures.
- End‑to‑end indexing → search → RAG pipelines using a lightweight mock LLM (OpenAI `gpt-3.5-turbo` with `temperature=0`).
- Service‑layer error handling (missing index, malformed config, etc.).

Run the full suite with coverage:

```bash
pytest -q --cov=mana_analyzer --cov-report=term-missing
```

### Continuous Integration

GitHub Actions execute the matrix:
- **Python 3.10‑3.13** on Ubuntu latest.
- Linting via **ruff** and type‑checking via **mypy**.
- `pre‑commit` hooks enforce black formatting, trailing‑whitespace removal, and doc‑string style.

The CI also builds the optional **FAISS‑GPU** wheel when a CUDA‑enabled runner is available.

### Local Development Tips

- Use the `scripts/reload.sh` helper to clear the index and rebuild from scratch:
  ```bash
  ./scripts/reload.sh /path/to/your/project
  ```
- The `MANAGER_DEBUG=1` env var enables very‑verbose logging of internal service calls.
- To experiment with a new LLM provider without altering source, create a `custom_llm.py` that subclasses `llm.base.LLMClient` and set `MANAGER_LLM_CLASS=custom_llm.MyClient`.

---

## Extending & Adding Language Support

### Adding a New Parser
1. Create a module under `src/mana_analyzer/parsers/` (e.g., `rust_parser.py`).
2. Implement two functions:
   - `def can_parse(path: Path) -> bool:` – return `True` for the file extensions you support.
   - `def parse(path: Path) -> list[Chunk]:` – read the source, optionally run a language‑specific lexer (tree‑sitter, rust‑ Analyzer), and emit `Chunk` objects.
3. Register the parser in `parsers/__init__.py` by adding it to the `SUPPORTED_PARSERS` list.
4. Write unit tests exercising edge‑cases such as multi‑line strings, macro definitions, and unusual comment styles.
5. Update the documentation section **[Extending & Adding Language Support](#extending)** with a brief example.

### Hooking a New Vector Store
If FAISS does not meet your scalability requirements, you can swap it for **Qdrant** or **Weaviate**:
- Implement `VectorStore` abstract base class (found in `services/vector_store.py`).
- Provide `upsert`, `search`, and `delete` methods matching FAISS semantics.
- Update `settings.Settings` with a `vector_store: Literal["faiss", "qdrant", "weaviate"]` field and a factory in `services/vector_store_factory.py`.
- The CLI automatically picks the configured store via the `--vector-store` flag.

---

## Advanced Configuration & Performance Tuning

| Setting | Description | Typical Values |
|--------|-------------|----------------|
| `MAX_WORKERS` (env) | Number of processes for parallel parsing/indexing. | `$(nproc)` or a fixed `8` |
| `CHUNK_TOKEN_LIMIT` | Maximum tokens per embedding chunk. | `400` (default) – raise to `800` for large monolithic files, at the cost of retrieval granularity. |
| `FAISS_METRIC` | Similarity metric (`inner_product` for cosine, `l2`). | `inner_product` |
| `LLM_TEMPERATURE` | Sampling temperature for LLM calls. | `0.0` for deterministic, `0.7` for creative. |
| `LLM_MAX_TOKENS` | Upper bound on generated token count per request. | `1024` |
| `CACHE_DIR` | Directory for downloaded model files (e.g., sentence‑transformers). | `~/.cache/mana_analyzer` |

#### Profiling Tips
- Use the built‑in `mana-analyzer profile` command to emit a `cProfile` report of the indexing pipeline.
- For massive monorepos, consider **sharding** the index: run `mana-analyzer index` separately on each sub‑project and later merge with `mana-analyzer merge-indexes` (future feature).
- Enable `MANAGER_LOG_LEVEL=DEBUG` to see per‑file hash calculations and skip‑logic for unchanged files.

---

## FAQ

**Q1: Why does `ask` sometimes return “I don’t know”?**
> The RAG pipeline only retrieves chunks that surpass the similarity threshold (`--score-threshold`). If no chunk is relevant, the LLM is prompted to admit uncertainty rather than hallucinate.

**Q2: Can I use a local LLM without an OpenAI API key?**
> Yes. Install a compatible HuggingFace model (e.g., `mistralai/Mistral-7B-Instruct-v0.2`) and set `OPENAI_BASE_URL=http://localhost:8000/v1` with an OpenAI‑compatible server like **vLLM** or **Text Generation Inference**.

**Q3: How does incremental indexing know a file changed?**
> Each file’s SHA‑256 hash is stored in the index metadata. During a subsequent run, the hash is recomputed; a mismatch triggers re‑embedding.

**Q4: My repository contains generated protobuf files – should I index them?**
> Generally no. Add a pattern to `--exclude "**/*.pb.go"` or edit `.manaignore` (similar to `.gitignore`).

**Q5: Is the tool safe for CI pipelines?**
> Absolutely. All operations are pure‑side‑effect‑free except for optional LLM calls. The CLI returns non‑zero exit codes on static‑analysis warnings when `--fail-on` is used, making it CI‑friendly.

---

## Contributing

Contributions are welcome!  Follow these steps:
1. Fork the repository and create a feature branch.
2. Ensure **pre‑commit** passes: `pre-commit run --all-files`.
3. Write tests for new functionality and achieve at least 90 % coverage.
4. Update documentation (README and any module‑level docstrings).
5. Submit a Pull Request with a clear description of the change and any relevant issue numbers.

Please read the `CODE_OF_CONDUCT.md` and `CONTRIBUTING.md` for detailed guidelines.

---

## License

`mana-analyzer` is released under the **MIT License**.  See the `LICENSE` file for full text.

---

## Acknowledgements

- The project builds on **LangChain**, **FAISS**, **Typer**, and **Rich** – huge thanks to their maintainers.
- Inspiration from open‑source RAG tools such as **Llama‑Index** and **Haystack**.
- Contributions from the community via issues, PRs, and model‑provider feedback.

---

## Contact & Support

- **GitHub Issues** – best place for bugs, feature requests, and usage questions.

Happy analyzing! 🎉
