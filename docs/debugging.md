# Debugging Guide

This page provides a quick debugging path for CLI/agent issues and flow-memory persistence.

## Fast triage checklist

1. Confirm runtime and environment:

```bash
python --version
which mana-analyzer
```

2. Run a narrow command with verbose logging:

```bash
mana-analyzer --verbose ask "what changed in flow memory?"
```

3. For coding-flow sessions, inspect active flow state:

```bash
mana-analyzer flow .
mana-analyzer flow . --format json
```

## Chat memory database location

`CodingMemoryService` persists per-project state to:

- `<project>/.mana_index/chat_memory.sqlite3`

Quick inspection:

```bash
sqlite3 .mana_index/chat_memory.sqlite3 ".tables"
sqlite3 .mana_index/chat_memory.sqlite3 "select flow_id,status,updated_at from coding_flows order by updated_at desc limit 10;"
```

## Chat memory schema (high-level)

Tables created by `CodingMemoryService._ensure_schema(...)`:

- `coding_flows`
- `coding_flow_turns`
- `coding_flow_decisions`
- `coding_flow_tasks`
- `coding_flow_checkpoints`

Key JSON columns:

- `coding_flows.constraints_json`
- `coding_flows.acceptance_json`
- `coding_flow_turns.changed_files_json`
- `coding_flow_turns.warnings_json`
- `coding_flow_turns.static_findings_json`
- `coding_flow_turns.checklist_json`
- `coding_flow_turns.transitions_json`

## Business rules enforced by `CodingMemoryService`

### Flow lifecycle

- `ensure_flow(...)` resumes an existing flow when possible; otherwise creates a new active flow with objective/constraint/acceptance extraction.
- `reset_flow(...)` marks flow status as `reset`.
- `checkpoint(...)` writes snapshot JSON into `coding_flow_checkpoints`.

### Turn persistence and derived data

- `record_turn(...)` writes full turn payloads and updates flow `updated_at`.
- `_extract_tasks(...)` parses markdown checkboxes into `coding_flow_tasks` (`done`/`open`).
- `_extract_decisions(...)` derives decisions from answer `Decision:` lines and warning heuristics.

### Patch-loop and conflict safeguards

- `has_prior_patch_failures(...)` checks recent warnings (last 3 turns) for retry-loop signals (`patch-style retry`, `patch-only loop`).
- `is_conflicting_request(...)` compares objective/request token overlap and edit intent.
  - Explicit plan-trigger requests matched by `_PLAN_TRIGGER_REQUEST_RE` are not treated as conflicts.

## Useful SQL snippets

Recent turns for a flow:

```sql
select created_at, user_request, warnings_json
from coding_flow_turns
where flow_id = '<flow_id>'
order by id desc
limit 10;
```

Open tasks for a flow:

```sql
select task_text, state, created_at
from coding_flow_tasks
where flow_id = '<flow_id>'
order by id desc
limit 50;
```

Recent decisions for a flow:

```sql
select decision, rationale, created_at
from coding_flow_decisions
where flow_id = '<flow_id>'
order by id desc
limit 20;
```

## Related docs

- [`coding-flows.md`](./coding-flows.md)
- [`optional-deps.md`](./optional-deps.md)
