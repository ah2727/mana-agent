# Coding Flows

## What a coding flow is

A coding flow is the persisted execution context for coding-agent turns in chat mode. It keeps:

- objective/constraints/acceptance signals extracted from user requests
- per-turn tool outcomes and warnings
- open/done checklist tasks
- recent decisions and transition history

Flows are project-scoped and stored in:

- `<project>/.mana_index/chat_memory.sqlite3`

## Stored data model

`CodingMemoryService` persists flow state into these SQLite tables:

- `coding_flows`: flow metadata (`flow_id`, objective, status, timestamps, constraints/acceptance JSON)
- `coding_flow_turns`: per-turn request/prompt/answer, changed files, warnings, static findings, checklist, transitions
- `coding_flow_tasks`: extracted checklist-like task rows (`open`/`done`)
- `coding_flow_decisions`: extracted decisions and rationales
- `coding_flow_checkpoints`: snapshots captured via checkpoint operations

`FlowSummary` is an aggregated read model over the latest flow + recent tasks/decisions/turns.

## How to inspect flow context

Top-level command:

```bash
mana-analyzer flow .
mana-analyzer flow . --format json
mana-analyzer flow . --flow-id <flow_id>
```

Chat commands:

- `/flow show`
- `/flow checklist`
- `/flow checkpoint`
- `/flow reset`

Database-level inspection (debugging):

```bash
sqlite3 .mana_index/chat_memory.sqlite3 ".tables"
sqlite3 .mana_index/chat_memory.sqlite3 "select flow_id, status, updated_at from coding_flows order by updated_at desc limit 5;"
```

## Planner/fallback and memory lifecycle

The coding agent and tools manager cooperate in this sequence:

1. Preview checklist generation:
   - `CodingAgent.preview_execution_checklist(...)` builds a pre-execution checklist.
   - Preview data is persisted with `CodingMemoryService.persist_preview_checklist(...)`.
2. Planner parse/repair/fallback:
   - Planner output is parsed.
   - If malformed, repair is attempted.
   - If still invalid, deterministic fallback checklist/plan is used.
3. Tool execution loop:
   - `ToolsManagerOrchestrator.run(...)` executes planner batches, tracks passes, warnings, and terminal reasons.
4. Transition and turn persistence:
   - `CodingAgent` records transitions/checklist outcomes.
   - `CodingMemoryService.record_turn(...)` stores turn payloads and task/decision extraction.
5. Flow control:
   - `checkpoint_flow(...)` writes snapshots.
   - `reset_flow(...)` marks flow status reset.

## Integration points

- [`src/mana_analyzer/llm/coding_agent.py`](../src/mana_analyzer/llm/coding_agent.py)
- [`src/mana_analyzer/llm/tools_manager.py`](../src/mana_analyzer/llm/tools_manager.py)
- [`src/mana_analyzer/services/coding_memory_service.py`](../src/mana_analyzer/services/coding_memory_service.py)
