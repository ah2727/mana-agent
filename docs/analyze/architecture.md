# Architecture Map

## Overview
High-level flow is split into layers that match the folder areas detected. The CLI/command layer (src/mana_agent/commands/*) defines the user-facing commands and routes them into the core services layer (src/mana_agent/services/*). The services layer coordinates the “what to do” logic and likely prepares context, including memory and repository-related information, before handing off to the LLM layer (src/mana_agent/llm/*).

The LLM layer appears to implement a “Live Agent Work Queue” via src/mana_agent/llm/agent_work_queue.py and related adapters/session/agent modules (agent_session.py, agent_work_queue_adapters.py, ask_agent.py, coding_agent.py, etc.). This layer connects to the tools layer (src/mana_agent/tools/*), which includes explicit repository and file edit primitives such as apply_patch.py, edit_file.py, write_file.py, and repository.py. The tools layer is what enables controlled mutations; tests reference specific “apply_patch” behaviors and mutation-plan/tool compilation (see tests/test_agent_work_queue.py).

Finally, the src/mana_agent/agent/* area provides orchestration/flow primitives (orchestrator.py, flow.py, evaluation_gate.py, task_classifier.py, selection.py, etc.), representing the shared “agent flow” machinery used by orchestration paths. Configuration is loaded via src/mana_agent/config/settings.py, while prompting is assembled via src/mana_agent/prompting/* and supported by utils and vector_store for retrieval/search behaviors. This layering supports safer testing because unit tests can validate queue/workflow logic and tool gating independently of the CLI.

## tests
Sample module docstring.

Related files:
- `tests/commands/test_analyze_slash_command.py`
- `tests/conftest.py`
- `tests/fixtures/sample_project/bad_module.py`
- `tests/fixtures/sample_project/good_module.py`
- `tests/fixtures/sample_project/no_doc.py`
- `tests/integration/test_chat_analyze_command.py`
- `tests/parsers/test_parser_adapters.py`
- `tests/test_agent_orchestrator.py`
- `tests/test_agent_work_queue.py`
- `tests/test_apply_patch_json_only.py`
- `tests/test_ask_agent.py`
- `tests/test_ask_agent_recovery.py`
- `tests/test_ask_service.py`
- `tests/test_ask_service_fallback.py`
- `tests/test_auto_chat.py`
- `tests/test_chat_background_index.py`
- `tests/test_chat_console_logging.py`
- `tests/test_chat_direct_commands.py`
- `tests/test_chat_planning_mode.py`
- `tests/test_checks.py`

Important symbols:
- `_FakeWorkerClient` `tests/test_agent_orchestrator.py:84`
- `_Local` `tests/test_agent_orchestrator.py:107`
- `_Redis` `tests/test_agent_orchestrator.py:111`
- `_AGENTIC_EDIT_TOOLS` `tests/test_agent_work_queue.py:31`
- `test_approved_mutation_plan_compiles_to_registered_tool` `tests/test_agent_work_queue.py:1059`
- `test_apply_patch_run_never_claims_no_edit_tool` `tests/test_agent_work_queue.py:1564`
- `_FakeExecutor` `tests/test_agent_work_queue.py:224`
- `_FakeExecutor` `tests/test_agent_work_queue.py:286`
- `_FakeWorker` `tests/test_agent_work_queue.py:411`
- `_NoDirectWorker` `tests/test_agent_work_queue.py:467`
- `_FakeExecutor` `tests/test_agent_work_queue.py:471`
- `_NoDirectWorker` `tests/test_agent_work_queue.py:540`
- `_FakeExecutor` `tests/test_agent_work_queue.py:544`
- `_FakeWorker` `tests/test_agent_work_queue.py:612`
- `_FakeWorker` `tests/test_agent_work_queue.py:690`
- `_TypoResolutionWorker` `tests/test_agent_work_queue.py:749`
- `_FakeWorker` `tests/test_agent_work_queue.py:818`
- `_ReadOnlyWorker` `tests/test_agent_work_queue.py:872`
- `_ProseOnlyWorker` `tests/test_agent_work_queue.py:926`
- `_ArchitectureWorker` `tests/test_agent_work_queue.py:979`

Dependencies on other parts:
- src/mana_agent/agent
- src/mana_agent/analysis
- src/mana_agent/commands
- src/mana_agent/config
- src/mana_agent/llm
- src/mana_agent/parsers
- src/mana_agent/prompting
- src/mana_agent/renderers
- src/mana_agent/services
- src/mana_agent/skills
- src/mana_agent/tools
- src/mana_agent/utils
- src/mana_agent/vector_store

Risk notes:
- Area spans many files; keep contracts documented and tested.

## src/mana_agent/services
mana_agent.services.ask_service

Related files:
- `src/mana_agent/services/__init__.py`
- `src/mana_agent/services/ask_service.py`
- `src/mana_agent/services/chat_service.py`
- `src/mana_agent/services/coding_memory_service.py`
- `src/mana_agent/services/coding_todo_service.py`
- `src/mana_agent/services/dependency_service.py`
- `src/mana_agent/services/describe_service.py`
- `src/mana_agent/services/index_service.py`
- `src/mana_agent/services/parsers/__init__.py`
- `src/mana_agent/services/parsers/base.py`
- `src/mana_agent/services/parsers/dart_parser.py`
- `src/mana_agent/services/parsers/js_ts_parser.py`
- `src/mana_agent/services/parsers/jvm_parser.py`
- `src/mana_agent/services/parsers/markup_parser.py`
- `src/mana_agent/services/parsers/native_parser.py`
- `src/mana_agent/services/parsers/python_parser.py`
- `src/mana_agent/services/parsers/scripting_parser.py`
- `src/mana_agent/services/project_analyze_service.py`
- `src/mana_agent/services/project_llm_analyze_service.py`
- `src/mana_agent/services/report_service.py`

Dependencies on other parts:
- src/mana_agent
- src/mana_agent/analysis
- src/mana_agent/config
- src/mana_agent/dependencies
- src/mana_agent/describe
- src/mana_agent/llm
- src/mana_agent/parsers
- src/mana_agent/utils
- src/mana_agent/vector_store

Risk notes:
- Area spans many files; keep contracts documented and tested.

## src/mana_agent/llm
Live Agent Work Queue.

Related files:
- `src/mana_agent/llm/__init__.py`
- `src/mana_agent/llm/agent_session.py`
- `src/mana_agent/llm/agent_work_queue.py`
- `src/mana_agent/llm/agent_work_queue_adapters.py`
- `src/mana_agent/llm/ask_agent.py`
- `src/mana_agent/llm/auto_chat.py`
- `src/mana_agent/llm/coding_agent.py`
- `src/mana_agent/llm/coding_agent_models.py`
- `src/mana_agent/llm/coding_agent_prompt.py`
- `src/mana_agent/llm/evidence_memory.py`
- `src/mana_agent/llm/gate_command.py`
- `src/mana_agent/llm/goal_profiles.py`
- `src/mana_agent/llm/mutation_plan.py`
- `src/mana_agent/llm/prompts.py`
- `src/mana_agent/llm/qna_chain.py`
- `src/mana_agent/llm/redis_tool_tasks.py`
- `src/mana_agent/llm/repo_chain.py`
- `src/mana_agent/llm/run_logger.py`
- `src/mana_agent/llm/small_direct_edit.py`
- `src/mana_agent/llm/tool_worker_process.py`

Dependencies on other parts:
- src/mana_agent/agent
- src/mana_agent/analysis
- src/mana_agent/commands
- src/mana_agent/config
- src/mana_agent/prompting
- src/mana_agent/services
- src/mana_agent/skills
- src/mana_agent/tools
- src/mana_agent/utils
- src/mana_agent/vector_store

Risk notes:
- Area spans many files; keep contracts documented and tested.

## src/mana_agent/agent
Agent flow primitives shared by ManaAgent orchestration paths.

Related files:
- `src/mana_agent/agent/__init__.py`
- `src/mana_agent/agent/evaluation_gate.py`
- `src/mana_agent/agent/evidence_queue.py`
- `src/mana_agent/agent/flow.py`
- `src/mana_agent/agent/orchestrator.py`
- `src/mana_agent/agent/selection.py`
- `src/mana_agent/agent/task_classifier.py`
- `src/mana_agent/agent/task_context.py`
- `src/mana_agent/agent/verification.py`
- `src/mana_agent/agent/verification_planner.py`

Dependencies on other parts:
- src/mana_agent/llm

## src/mana_agent/commands
Command package for mana-agent.

Related files:
- `src/mana_agent/commands/__init__.py`
- `src/mana_agent/commands/analyze_formats.py`
- `src/mana_agent/commands/chat_analyze_command.py`
- `src/mana_agent/commands/chat_cli.py`
- `src/mana_agent/commands/chat_input.py`
- `src/mana_agent/commands/cli.py`
- `src/mana_agent/commands/cli_internal.py`
- `src/mana_agent/commands/main_cli.py`
- `src/mana_agent/commands/output.py`
- `src/mana_agent/commands/ui_helpers.py`

Dependencies on other parts:
- src/mana_agent/analysis
- src/mana_agent/config
- src/mana_agent/dependencies
- src/mana_agent/describe
- src/mana_agent/llm
- src/mana_agent/parsers
- src/mana_agent/renderers
- src/mana_agent/services
- src/mana_agent/skills
- src/mana_agent/ui
- src/mana_agent/utils
- src/mana_agent/vector_store

## src/mana_agent/utils
mana_agent.utils.project_search

Related files:
- `src/mana_agent/utils/__init__.py`
- `src/mana_agent/utils/guards.py`
- `src/mana_agent/utils/index_discovery.py`
- `src/mana_agent/utils/io.py`
- `src/mana_agent/utils/logging.py`
- `src/mana_agent/utils/project_discovery.py`
- `src/mana_agent/utils/project_search.py`
- `src/mana_agent/utils/redaction.py`
- `src/mana_agent/utils/tool_policy.py`
- `src/mana_agent/utils/tools_run.py`

Dependencies on other parts:
- src/mana_agent/config

## src/mana_agent/prompting
Layered prompt construction for ManaAgent.

Related files:
- `src/mana_agent/prompting/__init__.py`
- `src/mana_agent/prompting/builder.py`
- `src/mana_agent/prompting/layers.py`
- `src/mana_agent/prompting/memory_snapshot.py`
- `src/mana_agent/prompting/mode_rules.py`
- `src/mana_agent/prompting/output_contract.py`
- `src/mana_agent/prompting/repo_rules.py`
- `src/mana_agent/prompting/skills_index.py`

Dependencies on other parts:
- src/mana_agent
- src/mana_agent/agent
- src/mana_agent/llm
- src/mana_agent/skills

## src/mana_agent/tools
mana_agent.tools

Related files:
- `src/mana_agent/tools/__init__.py`
- `src/mana_agent/tools/apply_patch.py`
- `src/mana_agent/tools/contracts.py`
- `src/mana_agent/tools/edit_file.py`
- `src/mana_agent/tools/repository.py`
- `src/mana_agent/tools/write_file.py`

## src/mana_agent/analysis
Static analysis and checks.

Related files:
- `src/mana_agent/analysis/__init__.py`
- `src/mana_agent/analysis/checks.py`
- `src/mana_agent/analysis/chunker.py`
- `src/mana_agent/analysis/models.py`

## src/mana_agent/parsers
Parsing logic.

Related files:
- `src/mana_agent/parsers/__init__.py`
- `src/mana_agent/parsers/multi_parser.py`
- `src/mana_agent/parsers/python_parser.py`

Dependencies on other parts:
- src/mana_agent/analysis

## src/mana_agent/vector_store
Embedding-client construction.

Related files:
- `src/mana_agent/vector_store/__init__.py`
- `src/mana_agent/vector_store/embeddings.py`
- `src/mana_agent/vector_store/faiss_store.py`

Dependencies on other parts:
- src/mana_agent/analysis
- src/mana_agent/config
- src/mana_agent/utils

## src/mana_agent
mana_agent package.

Related files:
- `src/mana_agent/__init__.py`
- `src/mana_agent/models.py`

## src/mana_agent/config
Configuration loading.

Related files:
- `src/mana_agent/config/__init__.py`
- `src/mana_agent/config/settings.py`

## src/mana_agent/dependencies
Dependency parsing / analysis.

Related files:
- `src/mana_agent/dependencies/__init__.py`
- `src/mana_agent/dependencies/dependency_service.py`

Dependencies on other parts:
- src/mana_agent
- src/mana_agent/analysis
- src/mana_agent/utils

## src/mana_agent/renderers
Rendering / output formatting.

Related files:
- `src/mana_agent/renderers/__init__.py`
- `src/mana_agent/renderers/html_report.py`

## src/mana_agent/skills
Modules under `src/mana_agent/skills`.

Related files:
- `src/mana_agent/skills/__init__.py`
- `src/mana_agent/skills/manager.py`

## src/mana_agent/ui
Terminal UI helpers for Mana Agent.

Related files:
- `src/mana_agent/ui/__init__.py`
- `src/mana_agent/ui/banner.py`

## src/mana_agent/default_skills
Built-in fallback skill templates.

Related files:
- `src/mana_agent/default_skills/__init__.py`

## src/mana_agent/describe
Description / summarization flows.

Related files:
- `src/mana_agent/describe/describe_service.py`

Dependencies on other parts:
- src/mana_agent/analysis
- src/mana_agent/dependencies
- src/mana_agent/utils

## Agent Workflow
- **Where is the command / CLI layer?** `src/mana_agent/commands/__init__.py`, `src/mana_agent/commands/analyze_formats.py`, `src/mana_agent/commands/chat_analyze_command.py`, `src/mana_agent/commands/chat_cli.py`, `src/mana_agent/commands/chat_input.py`
- **Where does the core application logic live?** `src/mana_agent/services/__init__.py`, `src/mana_agent/services/ask_service.py`, `src/mana_agent/services/chat_service.py`, `src/mana_agent/services/coding_memory_service.py`, `src/mana_agent/services/coding_todo_service.py`
- **Where is data modeled / persisted?** `src/mana_agent/vector_store/__init__.py`, `src/mana_agent/vector_store/embeddings.py`, `src/mana_agent/vector_store/faiss_store.py`
- **Where are external integrations?** `src/mana_agent/__init__.py`, `src/mana_agent/models.py`, `src/mana_agent/agent/__init__.py`, `src/mana_agent/agent/evaluation_gate.py`, `src/mana_agent/agent/evidence_queue.py`
- **Where is configuration loaded?** `src/mana_agent/config/__init__.py`, `src/mana_agent/config/settings.py`
- **Where are the tests?** `tests/commands/test_analyze_slash_command.py`, `tests/conftest.py`, `tests/fixtures/sample_project/bad_module.py`, `tests/fixtures/sample_project/good_module.py`, `tests/fixtures/sample_project/no_doc.py`
