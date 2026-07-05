# LLM Summary

_Model: gpt-5.4-nano_

## Project Summary
mana-agent is a Python CLI application that orchestrates LangChain-based “agent” workflows for asking, analyzing, and coding-related tasks. It includes a command layer (including chat-related commands) that routes into service and LLM/queue layers, plus a tools layer for repository/file mutations. The repository contains extensive tests around agent orchestration, work queues, and parsing/command behaviors.

## Detected Stack
The project is implemented in Python and uses LangChain (including langchain-community and langchain-openai), Typer for the CLI, and Pydantic/Pydantic Settings for configuration. It uses pip for dependency installation (no poetry.lock detected; a warning notes pip may still pin via requirements.txt). Runtime integrations detected include OpenAI, redis + rq, FAISS (faiss-cpu) for vector storage, and Rich for terminal output; tests use pytest.

## Architecture
High-level flow is split into layers that match the folder areas detected. The CLI/command layer (src/mana_agent/commands/*) defines the user-facing commands and routes them into the core services layer (src/mana_agent/services/*). The services layer coordinates the “what to do” logic and likely prepares context, including memory and repository-related information, before handing off to the LLM layer (src/mana_agent/llm/*).

The LLM layer appears to implement a “Live Agent Work Queue” via src/mana_agent/llm/agent_work_queue.py and related adapters/session/agent modules (agent_session.py, agent_work_queue_adapters.py, ask_agent.py, coding_agent.py, etc.). This layer connects to the tools layer (src/mana_agent/tools/*), which includes explicit repository and file edit primitives such as apply_patch.py, edit_file.py, write_file.py, and repository.py. The tools layer is what enables controlled mutations; tests reference specific “apply_patch” behaviors and mutation-plan/tool compilation (see tests/test_agent_work_queue.py).

Finally, the src/mana_agent/agent/* area provides orchestration/flow primitives (orchestrator.py, flow.py, evaluation_gate.py, task_classifier.py, selection.py, etc.), representing the shared “agent flow” machinery used by orchestration paths. Configuration is loaded via src/mana_agent/config/settings.py, while prompting is assembled via src/mana_agent/prompting/* and supported by utils and vector_store for retrieval/search behaviors. This layering supports safer testing because unit tests can validate queue/workflow logic and tool gating independently of the CLI.

## Agent Workflow
User requests are handled through the CLI/command layer (src/mana_agent/commands/*), which routes into the services layer (src/mana_agent/services/*) and then into the LLM/work-queue layer (src/mana_agent/llm/*). The work-queue/agent execution then invokes the tools layer (src/mana_agent/tools/*) for repository and file changes, with tests asserting safety properties around apply_patch and mutation-plan compilation (tests/test_agent_work_queue.py). A later coding-agent would patch the repository by following the tool contracts and ensuring test-gated behaviors continue to pass, then verify by running compilation and pytest.

## Developer Onboarding
Start by using the CLI entrypoint defined in pyproject.toml (mana_agent.commands.cli:app) to discover available command modes, then trace the call chain into src/mana_agent/services (e.g., ask_service.py) and further into src/mana_agent/llm (agent_work_queue.py / ask_agent.py). For changes that affect safety or file/repo mutations, focus on src/mana_agent/tools/* and run the specific work-queue tests (tests/test_agent_work_queue.py) because they encode important gating/behavior constraints. For reliability/safety, also review the existing tests that mock workers/executors and the risk-prone command strings in src/mana_agent/tools/repository.py and src/mana_agent/llm/ask_agent.py.
