# Improvement backlog (analysis)

This checklist captures short-term maintenance items surfaced while walking through
`mana-analyzer`. The focus areas mirror the current coding flow/agent tooling
(`src/mana_analyzer/commands`, `src/mana_analyzer/llm`, `src/mana_analyzer/services`) and
the observable lint/test gaps flagged while exploring the repo.

## 1. Flow persistence & heuristics

- `src/mana_analyzer/services/coding_memory_service.py` is currently undocumented:
  add module/class docstrings plus docstrings for the public helpers
  (`get_active_flow_id`, `ensure_flow`, `record_turn`, `build_flow_context`,
  `has_prior_patch_failures`, `is_conflicting_request`, etc.) so future maintainers can map
  each helper to the persisted schema and heuristics that guard flow continuity.
- Harden the `_extract_*` helpers with explicit unit tests that cover the heuristics
  (constraint/acceptance extraction, task parsing, decision detection, conflict detection) and
  add regression coverage for `has_prior_patch_failures` + `is_conflicting_request` to lock
  in the current guard rails around repeated patch-only retries and conflicting requests.
- Capture the `FlowSummary` and checklist shapes in a reusable dataclass helper so `build_flow_context`
  can be tested independently of the SQLite storage path (use temp dirs + fixtures).

## 2. CLI / coding-agent UX

- Many helpers in `src/mana_analyzer/commands/cli.py` (logging helpers, telemetry dataclasses,
  `RichToolCallbackHandler`, `_render_*` helpers) lack docstrings and are difficult to follow in code
  reviews. Document the flow of CLI telemetry + coding-agent turns, and extract smaller,
  testable functions where possible.
- Add higher-level tests for the CLI wrappers (commands such as `index`, `ask`, `chat`)
  that assert the logging/summary helpers render the expected data, ensuring the UI telemetry
  stays consistent when the agent output changes.

## 3. Documentation & troubleshooting bridge

- Keep `docs/coding-flows.md` aligned with the README "Coding Flows & Debugging" section
  whenever the schema, commands, or `FlowSummary` fields evolve. Consider referencing the
  latest schema field names (`open_tasks`, `recent_decisions`, `unresolved_static_findings`, etc.)
  in both locations so readers can switch between the CLI guidance and the deeper doc.
- Expand flow troubleshooting notes with real-world examples for stale active flows and
  conflicting requests. This helps teammates trace heuristics such as `_PLAN_TRIGGER_REQUEST_RE`
  or `has_prior_patch_failures` during debugging sessions.

## 4. Testing & static analysis coverage

- Introduce pytest fixtures that wrap the SQLite persistence (temporary `.mana_index`)
  so coding-memory/heavy CLI scenarios can be exercised without touching the user's home
  directory. Target the heuristics that guard plan-driven flows and conflicting edit sequences.
- Add coverage for the vector-store and search helpers (`FaissStore`, `analyze_service`,
  `ask_service`) to capture common failure modes (missing embeddings, API rate limits, etc.).
- Align the repo with the current ruff/mypy configuration by fixing the `unused-imports` errors
  emitted in `coding_memory_service.py` and any other services flagged by `ruff check src`.

## 5. Tooling polish

- Document the available extras (`dev`, `faiss-gpu`, `security`) in a dedicated `docs/optional-deps.md`
  that references the README section but includes compatibility notes and sample `pip install`
  commands.
- Provide a short `docs/debugging.md` snippet that lists the SQL schema for the chat memory
  database and the business rules enforced by `CodingMemoryService`, so contributors can
  debug flow persistence before launching the CLI.
