# Project Analysis Report

_Generated with LLM analysis (model: gpt-5.4-nano)._

## 1. Executive Summary
mana-agent is a Python CLI application that orchestrates LangChain-based “agent” workflows for asking, analyzing, and coding-related tasks. It includes a command layer (including chat-related commands) that routes into service and LLM/queue layers, plus a tools layer for repository/file mutations. The repository contains extensive tests around agent orchestration, work queues, and parsing/command behaviors.

## 2. Detected Stack
The project is implemented in Python and uses LangChain (including langchain-community and langchain-openai), Typer for the CLI, and Pydantic/Pydantic Settings for configuration. It uses pip for dependency installation (no poetry.lock detected; a warning notes pip may still pin via requirements.txt). Runtime integrations detected include OpenAI, redis + rq, FAISS (faiss-cpu) for vector storage, and Rich for terminal output; tests use pytest.

| Aspect | Detected |
| --- | --- |
| Languages | json, markdown, python, toml, yaml |
| Frameworks | LangChain, Typer |
| Package managers | pip |
| Testing | pytest |
| LLM / agent tooling | faiss-cpu, langchain, langchain-community, langchain-openai, openai |

| Metric | Value |
| --- | --- |
| Total Files | 219 |
| Source Files Count | 113 |
| Test Files Count | 60 |
| Config Files Count | 4 |
| Documentation Files Count | 38 |

## 3. Repository Overview
Source code lives under src/ and is organized by concerns: src/mana_agent/commands for CLI entrypoints and command implementations, src/mana_agent/services for core application logic (ask/chat/coding memory/todo/etc.), src/mana_agent/llm for the agent work queue and agent sessions, src/mana_agent/agent for shared flow/orchestration primitives, src/mana_agent/tools for repository/file mutation tools, and src/mana_agent/prompting/utils/vector_store/analysis/parsers for supporting functionality. Tests are under tests/ and include both unit and integration coverage across commands, analyzers, orchestrators, and work-queue/tool behavior.

Key folders:
- **source**: `src`
- **tests**: `tests`
- **docs**: `AGENTS.md`, `CHANGELOG.md`, `README.md`, `docs`, `src`

## 4. Important Files
| File | Why It Matters | Evidence |
| --- | --- | --- |
| `pyproject.toml` | Defines the CLI entrypoint that launches the Typer app. | Entrypoints evidence: name "mana-agent", type "cli", file "pyproject.toml" line 6, command "mana_agent.commands.cli:app". |
| `src/mana_agent/commands/cli.py` | CLI surface likely containing the Typer app referenced by pyproject.toml. | Architecture areas lists src/mana_agent/commands/cli.py under src/mana_agent/commands. |
| `src/mana_agent/services/ask_service.py` | Core logic for handling “ask” behaviors; central to chat/ask flows. | Architecture areas lists src/mana_agent/services/ask_service.py as mana_agent.services.ask_service. |
| `src/mana_agent/llm/agent_work_queue.py` | Implements the Live Agent Work Queue which drives agent execution scheduling/adapters. | Architecture areas lists src/mana_agent/llm/agent_work_queue.py under src/mana_agent/llm with responsibility "Live Agent Work Queue."; risks evidence references src/mana_agent/llm/agent_work_queue.py line 793. |
| `src/mana_agent/llm/ask_agent.py` | Implements the AskAgent behavior used by ask/chat workflows. | Architecture areas lists src/mana_agent/llm/ask_agent.py; risks evidence references unsafe shell commands in src/mana_agent/llm/ask_agent.py lines 149, 151. |
| `src/mana_agent/tools/repository.py` | Repository mutation operations; contains destructive command strings that must be guarded. | Risks evidence: src/mana_agent/tools/repository.py line 239 includes "rm -rf" and line 240 includes "git reset --hard". |
| `src/mana_agent/tools/apply_patch.py` | Patch application tool; critical to the tool-gating/mutation safety logic validated by tests. | Architecture areas lists src/mana_agent/tools/apply_patch.py; tests evidence includes test_apply_patch_run_never_claims_no_edit_tool in tests/test_agent_work_queue.py line 1564. |
| `tests/test_agent_orchestrator.py` | Tests orchestrator behavior and uses fake clients to validate flow correctness. | Important symbols evidence includes classes _FakeWorkerClient (line 84) and _Local/_Redis (lines 107/111) in tests/test_agent_orchestrator.py. |
| `tests/test_agent_work_queue.py` | Tests work-queue/tool selection and applies strong assertions about mutation-plan/tool compilation and apply_patch behavior. | Important symbols evidence includes _AGENTIC_EDIT_TOOLS (line 31) and tool-related tests such as test_approved_mutation_plan_compiles_to_registered_tool (line 1059) and test_apply_patch_run_never_claims_no_edit_tool (line 1564). |
| `.env.example` | Indicates configuration keys expected from environment; must not contain real secrets. | Secret-bearing config evidence lists ".env.example"; dependencies evidence includes python-dotenv and pydantic-settings. |

## 5. Entrypoints and Commands
The repository exposes a CLI named "mana-agent" via pyproject.toml entrypoint pointing to mana_agent.commands.cli:app. The commands package includes src/mana_agent/commands/chat_cli.py, chat_input.py, chat_analyze_command.py, and analyze_formats.py, plus cli_internal.py and main_cli.py, indicating multiple CLI modes and chat/analyze entrypoints. Tests also indicate an analyze slash command and integration coverage for chat analyze (e.g., tests/commands/test_analyze_slash_command.py and tests/integration/test_chat_analyze_command.py), but the exact runtime flags/options are not detected in the provided evidence.

| Name | Type | File | Line | Command |
| --- | --- | --- | ---: | --- |
| mana-agent | cli | `pyproject.toml` | 6 | `mana_agent.commands.cli:app` |

## 6. Architecture Map
High-level flow is split into layers that match the folder areas detected. The CLI/command layer (src/mana_agent/commands/*) defines the user-facing commands and routes them into the core services layer (src/mana_agent/services/*). The services layer coordinates the “what to do” logic and likely prepares context, including memory and repository-related information, before handing off to the LLM layer (src/mana_agent/llm/*).

The LLM layer appears to implement a “Live Agent Work Queue” via src/mana_agent/llm/agent_work_queue.py and related adapters/session/agent modules (agent_session.py, agent_work_queue_adapters.py, ask_agent.py, coding_agent.py, etc.). This layer connects to the tools layer (src/mana_agent/tools/*), which includes explicit repository and file edit primitives such as apply_patch.py, edit_file.py, write_file.py, and repository.py. The tools layer is what enables controlled mutations; tests reference specific “apply_patch” behaviors and mutation-plan/tool compilation (see tests/test_agent_work_queue.py).

Finally, the src/mana_agent/agent/* area provides orchestration/flow primitives (orchestrator.py, flow.py, evaluation_gate.py, task_classifier.py, selection.py, etc.), representing the shared “agent flow” machinery used by orchestration paths. Configuration is loaded via src/mana_agent/config/settings.py, while prompting is assembled via src/mana_agent/prompting/* and supported by utils and vector_store for retrieval/search behaviors. This layering supports safer testing because unit tests can validate queue/workflow logic and tool gating independently of the CLI.

### tests
Sample module docstring.
Related files: `tests/commands/test_analyze_slash_command.py`, `tests/conftest.py`, `tests/fixtures/sample_project/bad_module.py`, `tests/fixtures/sample_project/good_module.py`, `tests/fixtures/sample_project/no_doc.py`, `tests/integration/test_chat_analyze_command.py`, `tests/parsers/test_parser_adapters.py`, `tests/test_agent_orchestrator.py`
Risk notes: Area spans many files; keep contracts documented and tested.

### src/mana_agent/services
mana_agent.services.ask_service
Related files: `src/mana_agent/services/__init__.py`, `src/mana_agent/services/ask_service.py`, `src/mana_agent/services/chat_service.py`, `src/mana_agent/services/coding_memory_service.py`, `src/mana_agent/services/coding_todo_service.py`, `src/mana_agent/services/dependency_service.py`, `src/mana_agent/services/describe_service.py`, `src/mana_agent/services/index_service.py`
Risk notes: Area spans many files; keep contracts documented and tested.

### src/mana_agent/llm
Live Agent Work Queue.
Related files: `src/mana_agent/llm/__init__.py`, `src/mana_agent/llm/agent_session.py`, `src/mana_agent/llm/agent_work_queue.py`, `src/mana_agent/llm/agent_work_queue_adapters.py`, `src/mana_agent/llm/ask_agent.py`, `src/mana_agent/llm/auto_chat.py`, `src/mana_agent/llm/coding_agent.py`, `src/mana_agent/llm/coding_agent_models.py`
Risk notes: Area spans many files; keep contracts documented and tested.

### src/mana_agent/agent
Agent flow primitives shared by ManaAgent orchestration paths.
Related files: `src/mana_agent/agent/__init__.py`, `src/mana_agent/agent/evaluation_gate.py`, `src/mana_agent/agent/evidence_queue.py`, `src/mana_agent/agent/flow.py`, `src/mana_agent/agent/orchestrator.py`, `src/mana_agent/agent/selection.py`, `src/mana_agent/agent/task_classifier.py`, `src/mana_agent/agent/task_context.py`

### src/mana_agent/commands
Command package for mana-agent.
Related files: `src/mana_agent/commands/__init__.py`, `src/mana_agent/commands/analyze_formats.py`, `src/mana_agent/commands/chat_analyze_command.py`, `src/mana_agent/commands/chat_cli.py`, `src/mana_agent/commands/chat_input.py`, `src/mana_agent/commands/cli.py`, `src/mana_agent/commands/cli_internal.py`, `src/mana_agent/commands/main_cli.py`

### src/mana_agent/utils
mana_agent.utils.project_search
Related files: `src/mana_agent/utils/__init__.py`, `src/mana_agent/utils/guards.py`, `src/mana_agent/utils/index_discovery.py`, `src/mana_agent/utils/io.py`, `src/mana_agent/utils/logging.py`, `src/mana_agent/utils/project_discovery.py`, `src/mana_agent/utils/project_search.py`, `src/mana_agent/utils/redaction.py`

### src/mana_agent/prompting
Layered prompt construction for ManaAgent.
Related files: `src/mana_agent/prompting/__init__.py`, `src/mana_agent/prompting/builder.py`, `src/mana_agent/prompting/layers.py`, `src/mana_agent/prompting/memory_snapshot.py`, `src/mana_agent/prompting/mode_rules.py`, `src/mana_agent/prompting/output_contract.py`, `src/mana_agent/prompting/repo_rules.py`, `src/mana_agent/prompting/skills_index.py`

### src/mana_agent/tools
mana_agent.tools
Related files: `src/mana_agent/tools/__init__.py`, `src/mana_agent/tools/apply_patch.py`, `src/mana_agent/tools/contracts.py`, `src/mana_agent/tools/edit_file.py`, `src/mana_agent/tools/repository.py`, `src/mana_agent/tools/write_file.py`

### src/mana_agent/analysis
Static analysis and checks.
Related files: `src/mana_agent/analysis/__init__.py`, `src/mana_agent/analysis/checks.py`, `src/mana_agent/analysis/chunker.py`, `src/mana_agent/analysis/models.py`

### src/mana_agent/parsers
Parsing logic.
Related files: `src/mana_agent/parsers/__init__.py`, `src/mana_agent/parsers/multi_parser.py`, `src/mana_agent/parsers/python_parser.py`

### src/mana_agent/vector_store
Embedding-client construction.
Related files: `src/mana_agent/vector_store/__init__.py`, `src/mana_agent/vector_store/embeddings.py`, `src/mana_agent/vector_store/faiss_store.py`

### src/mana_agent
mana_agent package.
Related files: `src/mana_agent/__init__.py`, `src/mana_agent/models.py`

### src/mana_agent/config
Configuration loading.
Related files: `src/mana_agent/config/__init__.py`, `src/mana_agent/config/settings.py`

### src/mana_agent/dependencies
Dependency parsing / analysis.
Related files: `src/mana_agent/dependencies/__init__.py`, `src/mana_agent/dependencies/dependency_service.py`

### src/mana_agent/renderers
Rendering / output formatting.
Related files: `src/mana_agent/renderers/__init__.py`, `src/mana_agent/renderers/html_report.py`

### src/mana_agent/skills
Modules under `src/mana_agent/skills`.
Related files: `src/mana_agent/skills/__init__.py`, `src/mana_agent/skills/manager.py`

### src/mana_agent/ui
Terminal UI helpers for Mana Agent.
Related files: `src/mana_agent/ui/__init__.py`, `src/mana_agent/ui/banner.py`

### src/mana_agent/default_skills
Built-in fallback skill templates.
Related files: `src/mana_agent/default_skills/__init__.py`

### src/mana_agent/describe
Description / summarization flows.
Related files: `src/mana_agent/describe/describe_service.py`

## 7. Agent Workflow
User requests are handled through the CLI/command layer (src/mana_agent/commands/*), which routes into the services layer (src/mana_agent/services/*) and then into the LLM/work-queue layer (src/mana_agent/llm/*). The work-queue/agent execution then invokes the tools layer (src/mana_agent/tools/*) for repository and file changes, with tests asserting safety properties around apply_patch and mutation-plan compilation (tests/test_agent_work_queue.py). A later coding-agent would patch the repository by following the tool contracts and ensuring test-gated behaviors continue to pass, then verify by running compilation and pytest.

- **Where is the command / CLI layer?** `src/mana_agent/commands/__init__.py`, `src/mana_agent/commands/analyze_formats.py`, `src/mana_agent/commands/chat_analyze_command.py`, `src/mana_agent/commands/chat_cli.py`, `src/mana_agent/commands/chat_input.py`
- **Where does the core application logic live?** `src/mana_agent/services/__init__.py`, `src/mana_agent/services/ask_service.py`, `src/mana_agent/services/chat_service.py`, `src/mana_agent/services/coding_memory_service.py`, `src/mana_agent/services/coding_todo_service.py`
- **Where is data modeled / persisted?** `src/mana_agent/vector_store/__init__.py`, `src/mana_agent/vector_store/embeddings.py`, `src/mana_agent/vector_store/faiss_store.py`
- **Where are external integrations?** `src/mana_agent/__init__.py`, `src/mana_agent/models.py`, `src/mana_agent/agent/__init__.py`, `src/mana_agent/agent/evaluation_gate.py`, `src/mana_agent/agent/evidence_queue.py`
- **Where is configuration loaded?** `src/mana_agent/config/__init__.py`, `src/mana_agent/config/settings.py`
- **Where are the tests?** `tests/commands/test_analyze_slash_command.py`, `tests/conftest.py`, `tests/fixtures/sample_project/bad_module.py`, `tests/fixtures/sample_project/good_module.py`, `tests/fixtures/sample_project/no_doc.py`

## 8. Analyze Workflow
For chat-integrated repository analysis, the flow appears to involve analyze-related commands in src/mana_agent/commands (chat_analyze_command.py and analyze_formats.py) and integration tests in tests/integration/test_chat_analyze_command.py. The system likely uses the services layer and prompting/analyzer components under src/mana_agent/analysis and src/mana_agent/prompting to produce the analysis output; however, the exact /analyze endpoint mapping is not detected. The evidence indicates the analyze slash command is tested at tests/commands/test_analyze_slash_command.py.

## 9. Dependencies
- Runtime: faiss-cpu, langchain, langchain-community, langchain-openai, openai, prompt_toolkit, pydantic, pydantic-settings, python-dotenv, redis, rich, rq, safety, tenacity, typer
- Dev: pytest
- Lock files: none detected
- ⚠️ Python dependency manifest detected without poetry.lock; pip may still use requirements pinning.

## 10. Symbols Overview
Tests define multiple fake worker/executor classes to simulate worker behavior and validate orchestration logic: _FakeWorkerClient, _Local, _Redis in tests/test_agent_orchestrator.py. In work-queue tests, there is an explicit tool list constant _AGENTIC_EDIT_TOOLS (tests/test_agent_work_queue.py line 31), and critical test functions enforce safety and compilation rules such as test_approved_mutation_plan_compiles_to_registered_tool (line 1059) and test_apply_patch_run_never_claims_no_edit_tool (line 1564). Chat-related tests include fakes for search and LLM messages (tests/test_ask_agent.py symbols like _FakeSearchService, _FakeAIMessage, _FakeLLM), used to assert planning and tool invocation/dedup behaviors.

- Python files scanned: 173
- Symbols extracted: 3185
- `_FakeWorkerClient` (class) `tests/test_agent_orchestrator.py:84`
- `_Local` (class) `tests/test_agent_orchestrator.py:107`
- `_Redis` (class) `tests/test_agent_orchestrator.py:111`
- `_AGENTIC_EDIT_TOOLS` (tool) `tests/test_agent_work_queue.py:31`
- `test_approved_mutation_plan_compiles_to_registered_tool` (tool) `tests/test_agent_work_queue.py:1059`
- `test_apply_patch_run_never_claims_no_edit_tool` (tool) `tests/test_agent_work_queue.py:1564`
- `_FakeExecutor` (class) `tests/test_agent_work_queue.py:224`
- `_FakeExecutor` (class) `tests/test_agent_work_queue.py:286`
- `_FakeWorker` (class) `tests/test_agent_work_queue.py:411`
- `_NoDirectWorker` (class) `tests/test_agent_work_queue.py:467`
- `_FakeExecutor` (class) `tests/test_agent_work_queue.py:471`
- `_NoDirectWorker` (class) `tests/test_agent_work_queue.py:540`
- `_FakeExecutor` (class) `tests/test_agent_work_queue.py:544`
- `_FakeWorker` (class) `tests/test_agent_work_queue.py:612`
- `_FakeWorker` (class) `tests/test_agent_work_queue.py:690`
- `_TypoResolutionWorker` (class) `tests/test_agent_work_queue.py:749`
- `_FakeWorker` (class) `tests/test_agent_work_queue.py:818`
- `_ReadOnlyWorker` (class) `tests/test_agent_work_queue.py:872`
- `_ProseOnlyWorker` (class) `tests/test_agent_work_queue.py:926`
- `_ArchitectureWorker` (class) `tests/test_agent_work_queue.py:979`
- `_ShouldNotRunWorker` (class) `tests/test_agent_work_queue.py:1096`
- `_ShouldNotRunWorker` (class) `tests/test_agent_work_queue.py:1133`
- `_StructuredCommandWorker` (class) `tests/test_agent_work_queue.py:1163`
- `_ProseWorker` (class) `tests/test_agent_work_queue.py:1213`
- `_ReadOnlyWorker` (class) `tests/test_agent_work_queue.py:1259`

## 11. Risks and Problems
- **High** Unsafe shell commands present in repository/tooling — repository
  - Evidence: src/mana_agent/tools/repository.py lines 239-240 include command strings "rm -rf" and "git reset --hard"; risks evidence also mentions unsafe shell command in src/mana_agent/llm/ask_agent.py lines 149 and 151 for "rm -rf" and "git reset --hard".
  - Why it matters: Destructive commands can remove or reset user work if unguarded, parameterized incorrectly, or executed unexpectedly during agent runs.
  - Recommended fix: Ensure destructive commands are blocked or require explicit, user-visible confirmation and are restricted to narrowly-scoped, validated paths/operations. Add/extend tests verifying that only allowed operations execute given specific worker/tool outcomes.
- **High** Unsafe rm -rf invocation demonstrated in tests — repository
  - Evidence: tests/test_ask_agent.py line 295 uses run_command.invoke({"cmd": "rm -rf /tmp/foo"}).
  - Why it matters: Even if used as a negative/guard test, it indicates the system accepts raw command input paths where rm -rf could be attempted, so safety gating must be reliable.
  - Recommended fix: Tighten command execution validation: reject dangerous commands at parsing time and add explicit assertions that the dangerous command never executes.
- **High** Secrets exposure risk in tracked docs/tests/code — repository
  - Evidence: Multiple files contain redacted but present patterns like "OPENAI_<redacted>" or openai_<redacted>" in CHANGELOG.md (line 154), README.md (line 116), and many tests (e.g., tests/test_chat_planning_mode.py line 14, tests/test_cli_smoke.py line 174) plus docs/03-quick-start.md (line 36). Also CHANGELOG.md line 17 references restored rm -rf blocking behavior in AskAgent.run_command.
  - Why it matters: If real credentials exist in tracked files, they can be leaked via repository history and logs; this is especially critical for LLM providers.
  - Recommended fix: Purge any committed secrets from history (not just working tree), ensure runtime uses environment variables loaded from .env.example, and rotate any affected credentials. Add a CI guard that fails on patterns matching known secret prefixes.

Static-analysis findings (from the merged static engine):
- `missing-docstring`: 1618
- `deep-nesting`: 313
- `unused-imports`: 205
- `wildcard-import`: 8

## 12. Recommendations
- Clarify dependency lock policy by documenting the intended pip workflow between pyproject.toml and requirements.txt, and ensure CI consistently installs from the chosen files (evidence: dependency warning about missing poetry.lock).
- Add/verify timeouts around any long-running subprocess/command execution helpers used by CLI modes and agent work queue paths (evidence: recommendation mentions specific files, but no direct timeout evidence was detected).
- Ensure agent context artifacts (e.g., .mana/analyze/agent_context.json) are loaded for chat context grounding and never include secrets (evidence: recommendation states this artifact exists, but the artifact path is not detected in the file listing; treat as not detected and verify before implementing).

## 13. Next Coding Tasks
### Block/guard destructive repository operations with explicit confirmation and safe targeting
- Priority: High
- Files likely involved: `src/mana_agent/tools/repository.py`, `src/mana_agent/llm/ask_agent.py`
- Acceptance criteria:
  - No rm -rf or git reset --hard commands execute without explicit, user-confirmed intent and strict path validation (evidence-based scope tied to the detected command strings).
  - Existing tests continue to pass, and new tests (if added) assert that unsafe commands are rejected/blocked.
- Verification: `pytest -q`

### Remove any committed secrets from tracked files and ensure tests/docs never write real credentials
- Priority: High
- Files likely involved: `README.md`, `CHANGELOG.md`, `docs/03-quick-start.md`, `tests/test_chat_planning_mode.py`, `tests/test_cli_smoke.py`, `tests/test_ask_agent_recovery.py`, `tests/test_tool_worker_process.py`
- Acceptance criteria:
  - Replace secret-bearing values with environment-variable references and ensure .env.example contains only non-secret placeholders.
  - Add/enable a repository-level check that fails if secret patterns (e.g., OPENAI_/openai_) appear in tracked files.
- Verification: `python -m compileall . && pytest -q`

### Stabilize test suite and CI by aligning dependency installation policy
- Priority: Medium
- Files likely involved: `pyproject.toml`, `requirements.txt`
- Acceptance criteria:
  - Document whether pip installs from requirements.txt or from pyproject.toml and how version pinning is enforced.
  - CI installs deterministically and compilation succeeds.
- Verification: `python -m compileall src`

## 14. Generated Artifacts
| Artifact | Purpose |
| --- | --- |
| `.mana/analyze/report.md` | Human-readable senior-engineer report (this file). |
| `.mana/analyze/report.json` | Full machine-readable report: deterministic data + LLM analysis. |
| `.mana/analyze/agent_context.json` | Compact, bounded context loaded into chat/coding-agent turns. |
| `.mana/analyze/evidence.json` | Compact structured evidence used as input to the LLM analyzer. |
| `.mana/analyze/llm_summary.md` | LLM-written narrative summary and onboarding notes. |
| `.mana/analyze/inventory.json` | File inventory, classifications, folders, and counts. |
| `.mana/analyze/symbols.json` | Extracted Python classes/functions/commands with locations. |
| `.mana/analyze/dependencies.json` | Runtime/dev dependencies, lock files, and tooling packages. |
| `.mana/analyze/architecture.md` | Architecture map with layers, files, and LLM explanation. |
| `.mana/analyze/risks.json` | Detected risks with severity, evidence, and fixes. |
| `.mana/analyze/recommendations.md` | Prioritized recommendations and next coding tasks. |
