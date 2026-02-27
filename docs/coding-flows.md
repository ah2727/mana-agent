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

Common `FlowSummary` fields surfaced by `mana-analyzer flow --format json`:

- `objective`
- `open_tasks`
- `recent_decisions`
- `last_changed_files`
- `unresolved_static_findings`
- `checklist`
- `transitions`
- `last_blocked_reason`

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

## Troubleshooting examples

### Stale active flow keeps resurfacing old tasks

Symptoms:

- `mana-analyzer flow .` still shows outdated `open_tasks`
- new requests appear to inherit old checklist context

Checks:

```bash
mana-analyzer flow . --format json
sqlite3 .mana_index/chat_memory.sqlite3 "select flow_id,status,updated_at from coding_flows order by updated_at desc limit 10;"
```

Actions:

- run `/flow reset` in chat or start a new flow for a divergent request
- verify no automation/test fixture is reusing the same `flow_id` unintentionally

### New edit request is flagged as conflicting

Symptoms:

- chat prompts for `continue` vs `new` flow choice
- request is considered off-track from current objective

Why:

- `CodingMemoryService.is_conflicting_request(...)` compares current objective words with request words and treats
  low-overlap edit-intent requests as a track switch
- explicit plan trigger requests (`implement plan`) are ignored by `_PLAN_TRIGGER_REQUEST_RE`

Actions:

- reply `continue` if request is same track but phrased differently
- reply `new` if request intentionally starts a separate task stream

## Integration points

- [`src/mana_analyzer/llm/coding_agent.py`](../src/mana_analyzer/llm/coding_agent.py)
- [`src/mana_analyzer/llm/tools_manager.py`](../src/mana_analyzer/llm/tools_manager.py)
- [`src/mana_analyzer/services/coding_memory_service.py`](../src/mana_analyzer/services/coding_memory_service.py)
