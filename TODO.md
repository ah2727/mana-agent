

## 1️⃣ Documentation & Public API

| # | Action | Where | Why |
|---|--------|-------|-----|
| 1 | **Add module‑level docstrings** to every package file (e.g. `src/mana_analyzer/services/chat_service.py`, `src/mana_analyzer/services/ask_service.py`, 
`src/mana_analyzer/llm/repo_chain.py`). The files currently only have empty docstrings or none at all【source: chat_service.py:1‑13】【source: 
ask_service.py:1‑13】【source: repo_chain.py:1‑13】. | Improves discoverability for users and for generated API docs. |
| 2 | **Document the public classes & methods** (`ChatService`, `AskService`, `SearchService.search_multi`, `RepositoryMultiChain.summarize_file`, etc.) with 
clear parameter/return descriptions. | The methods have no `"""…"""` text, making it hard for contributors and for tools like Sphinx or IDEs. |
| 3 | **Add a “TODO” section in the README** that outlines current limitations (e.g., no async support yet, missing auto‑indexing for dir‑mode) so contributors
can see open work at a glance. | The README currently only shows usage; a road‑map helps drive community contributions. |
| 4 | **Document the expected JSON schema for tool‑friendly outputs** (`tool_semantic_search`, `tool_semantic_search_multi`) in the README or a separate docs 
file. | Users may want to call these from external agents; explicit schema prevents misuse. |

---

## 2️⃣ Type‑Hints & Signatures

| # | Action | Where | Why |
|---|--------|-------|-----|
| 5 | **Add missing return type hints** for functions that currently lack them, e.g. `AskService.ask_dir_mode_with_tools` (ends with `):` but no `-> 
AskResponse`). | Improves static analysis, mypy compliance, and IDE autocompletion. |
| 6 | **Annotate `ChatService.__init__` parameters** (`settings`, `model_override`, `index_dir`, etc.) with concrete types (`Settings`, `str | None`, …). 
Currently the method signature is untyped beyond the `*` placeholders【source: chat_service.py:23‑71】. |
| 7 | **Add explicit type hints to every public helper** (`_resolve_out_path`, `_index_has_vectors`, `_index_has_chunks`, etc.) – they already return 
`Path`/`bool` but the signature should be declared. |
| 8 | **Make `AskCallback` a concrete Protocol** – it already specifies `on_event`, but add a return type (`-> None`) and consider adding a `__call__` overload
for ease of use. |
| 9 | **Add generic `Sequence[Any]` vs `list[Any]` where appropriate** – e.g., `AskService.ask_with_tools` accepts `callbacks: Sequence[Any] | None`; the same 
pattern should be used consistently. |

---

## 3️⃣ Error Handling & Edge Cases

| # | Action | Where | Why |
|---|--------|-------|-----|
|10| **Guard against missing index directories** in `SearchService.search`. It already falls back to lexical search, but if *both* approaches fail the user 
gets an empty list with no explicit error. Add a clear `RuntimeError` or return a diagnostic object. | `SearchService.search` currently returns `[]` 
silently【source: search_service.py:24‑35】. |
|11| **Validate `k` parameters** – many methods assume `k > 0` (e.g., `AskService.ask`, `SearchService.search_multi`) but don’t enforce it. Raise `ValueError` 
for non‑positive values. |
|12| **Handle malformed JSON in tool wrappers** – `SearchService.tool_semantic_search`/`tool_semantic_search_multi` assume the underlying store returns 
well‑formed data. Wrap JSON serialization in try/except and surface a helpful message. |
|13| **Fix a subtle bug in `_group_sources_by_index`** – the condition `if matched is None or len(str(subproject_root)) > len(str(matched.parent)):` may raise 
an `AttributeError` when `matched` is `None`. The intention is to compare lengths **after** `matched` is set. Refactor the logic to avoid accessing 
`matched.parent` when `matched` is `None`. (The code lives in `AskService._group_sources_by_index`【source: ask_service.py:115‑137】). |
|14| **Make `ChatService.ask` robust** – it swallows all exceptions when appending history (`except Exception: pass`). Log the exception instead of silently 
ignoring it. |
|15| **Add explicit “index missing” warnings** for dir‑mode (`AskService.ask_dir_mode`) when no indexes are found – currently it returns a warning but also 
adds it to the `warnings` list; clarify the contract and maybe raise a custom `MissingIndexError`. |

---

## 4️⃣ Test Coverage & Quality

| # | Action | Where | Why |
|---|--------|-------|-----|
|16| **Write unit tests for `SearchService._lexical_search`** – currently there are no tests covering the fallback token‑based search. |
|17| **Add tests for error paths** – e.g., passing a non‑existent index directory, negative `k`, or malformed JSON payloads. |
|18| **Increase coverage for `RepositoryMultiChain`** – only a few public methods (`summarize_file`, `synthesize_deep_flow_analysis`) are exercised. Include 
tests for `_compact_dependency_report` and `_compact_file_summaries`. |
|19| **Test the new return‑type annotations** after they’re added (mypy should pass). |
|20| **Add integration tests for the CLI** (`mana-analyzer`) using `typer.testing.CliRunner` to ensure commands like `index`, `search`, and `ask` behave as 
expected under error conditions. |
|21| **Add a `tests/fixtures/` directory** with small sample repos (Python, JS, etc.) to be used by multiple test suites (search, static analysis, dependency 
graph). |
|22| **Add a test for `ChatService.__init__`** ensuring that `settings` are actually respected (e.g., default `index_dir`). |

---

## 5️⃣ Logging Consistency & Observability

| # | Action | Where | Why |
|---|--------|-------|-----|
|23| **Standardise log‑field names** – some logs use `index_dir=` while others embed the path in the message. Adopt a structured approach (`extra={"index_dir":
str(index_dir)}`) across the codebase. |
|24| **Expose the logger via the package** – provide `mana_analyzer.utils.logging.get_logger(name)` instead of importing `logging` directly, to centralise 
formatting. |
|25| **Add log rotation for `.mana_logs`** – currently logs may grow indefinitely. Use `logging.handlers.RotatingFileHandler`. |
|26| **Add correlation IDs** for a request (question) that flow through `AskService`, `AskAgent`, and `SearchService` (e.g., a UUID in `extra`). This helps 
trace multi‑step RAG runs. |
|27| **Log warnings when falling back from vector to lexical search** (already an info, but consider a `warning` level to surface potential indexing problems).
|

---

## 6️⃣ Performance & Parallelism

| # | Action | Where | Why |
|---|--------|-------|-----|
|28| **Make `SearchService.search_multi` fully async** – currently it uses a `ThreadPoolExecutor`. Switching to `asyncio` (with `anyio` or `trio`) would reduce
thread overhead on I/O‑bound workloads. |
|29| **Profile the chunking pipeline** (`CodeChunker.build_chunks`) for large repos and consider streaming chunks to avoid loading everything in memory. |
|30| **Cache `SearchService._tokenize` results** (e.g., via `functools.lru_cache`) because tokenization of the same query may happen repeatedly in multi‑index 
searches. |
|31| **Add a `--max-workers` CLI flag** to let users tune the concurrency level for indexing, searching, and static analysis. |
|32| **Avoid repeated `Path.resolve()`** – many functions resolve the same path many times; compute once and reuse to reduce filesystem calls. |

---

## 7️⃣ API & CLI Enhancements

| # | Action | Where | Why |
|---|--------|-------|-----|
|33| **Add `--json` flag to all commands** (already present for many, but ensure `chat`, `ask`, and `describe` also output machine‑readable JSON). |
|34| **Introduce an `--dry-run` option** for `index` and `describe` to preview actions without writing to disk. |
|35| **Implement auto‑indexing in directory‑mode** – when `ask_dir_mode` detects a missing `.mana_index`, automatically create a temporary index (using the 
`_make_ephemeral_index_dir` helper) and clean it up after the request. |
|36| **Expose a `mana-analyzer version` sub‑command** that prints the package version and git commit hash (use `importlib.metadata.version`). |
|37| **Add shell‑completion scripts** (`bash`, `zsh`, `fish`) via `typer`’s builtin support. |
|38| **Add a `--config` option** to point to an alternative `.env` file, making the CLI more flexible in CI pipelines. |
|39| **Add a `--profile` flag** that writes a `cProfile` report for any command, useful for performance debugging. |

---

## 8️⃣ CI / Packaging / Release

| # | Action | Where | Why |
|---|--------|-------|-----|
|40| **Add a GitHub Actions workflow** that runs `pytest`, `mypy`, `ruff` (or `flake8`) and checks that the generated documentation builds (`mkdocs` or 
`sphinx`). |
|41| **Add a `pre‑commit` configuration** with hooks for black, isort, ruff, and mypy. |
|42| **Update `pyproject.toml`** to include a proper `scripts` entry (`mana-analyzer = "mana_analyzer.commands.cli:app"` is already there, but add 
`requires-python = ">=3.10,<3.15"` to match the supported range). |
|43| **Publish wheels to PyPI** for the three major platforms (linux, macOS, windows) via `cibuildwheel`. |
|44| **Add a `CODE_OF_CONDUCT.md`** and `CONTRIBUTING.md` to guide new contributors. |
|45| **Add a `SECURITY.md`** to specify how to report vulnerabilities (especially since the code calls `safety`). |
|46| **Pin transitive dependencies** (e.g., `langchain`, `openai`) in a `constraints.txt` for reproducible builds. |
|47| **Add `setup.cfg` or `pyproject.toml` metadata** for long description, classifiers, and project URLs (homepage, bug tracker, docs). |

---

## 9️⃣ Refactoring Opportunities

| # | Action | Where | Why |
|---|--------|-------|-----|
|48| **Extract duplicate logging helpers** (`_log_call`, `_log_return`, `_log_exception`) into a shared `utils.logging_helpers` module – they are currently 
defined in `cli.py` and duplicated in other modules. |
|49| **Consolidate index discovery** – both `cli` and `ChatService` perform index discovery (`discover_index_dirs`). Create a single public helper 
(`mana_analyzer.utils.index_discovery.discover_all_indexes`). |
|50| **Rename `AskService.ask_dir_mode_with_tools`** to a clearer `ask_dir_mode_with_tools` (follow snake_case like the other methods). |
|51| **Move large prompt strings** (`FILE_SUMMARY_SYSTEM`, `ARCH_SYSTEM`, etc.) into a dedicated `prompts.py` module (they already live there, but some prompts
are duplicated in `repo_chain.py`; keep a single source). |
|52| **Replace manual string concatenation** in `SearchService._render_context` with `"\n".join([...])` for readability. |
|53| **Introduce a `BaseService` class** that provides `_resolve_report_artifact_dir` and common logger config, reducing boilerplate across `ReportService`, 
`StructureService`, etc. |
|54| **Add `__all__` declarations** for public sub‑modules (e.g., `src/mana_analyzer/services/__init__.py`) to make the public API explicit. |

---

## 10️⃣ Miscellaneous

| # | Action | Where | Why |
|---|--------|-------|-----|
|55| **Fix a missing import** – `from typing import List, Optional, Union` is present in `chat_service.py` but `List` is not used; remove or use. |
|56| **Add a `requirements-dev.txt`** that mirrors `pyproject.toml` dev dependencies (`pytest`, `pytest‑mock`, `pre‑commit`, etc.) for contributors who prefer 
`pip install -r`. |
|57| **Update README badges** (PyPI version, CI status, coverage) for better visibility. |
|58| **Add examples of the JSON output** for `ask` and `search` commands, so users can see the exact schema. |
|59| **Document environment variables** (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `MANA_LOG_LEVEL`) in the README. |
|60| **Implement a `--quiet` flag** that suppresses Rich console output for scripted use. |

---

### How to use the checklist

1. **Start with low‑effort wins**: add docstrings, type hints, and basic error handling (items 1‑3).  
2. **Run the test suite** and add the missing coverage (items 16‑22).  
3. **Upgrade logging** and CI (items 23‑28, 40‑45).  
4. **Address the functional bugs** (`_group_sources_by_index`, missing returns) before large refactors.  
5. **Iterate on performance & CLI UX** (items 28‑39).  

Following this roadmap will make the repository:

* **More developer‑friendly** (clear docs, strict types, full test coverage).  
* **More reliable** (robust error handling, better logging, CI checks).  
* **More performant & scalable** (parallel/async searches, smarter caching).  
* **Easier to adopt** (CLI flags, completion scripts, packaging).  

Feel free to pick any subset that matches your team’s capacity—each item is independent and provides immediate value. Happy coding!