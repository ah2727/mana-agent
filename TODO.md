# Improvement backlog

## Coding memory service hygiene

- Add a module-level docstring plus docstrings for `FlowSummary`, `CodingMemoryService`, and the public helpers
  (`get_active_flow_id`, `ensure_flow`, `record_turn`, `build_flow_context`, `is_conflicting_request`, etc.)
  so folks can quickly understand how flows/turns/tasks are persisted and why the current heuristics exist. Ruff
  already flags missing docstrings here.
- Double-check the `from __future__ import annotations` line and any other unused imports in
  `src/mana_analyzer/services/coding_memory_service.py` so lint warnings about unused names go away without
  regressing typing support.

## Testing and behavior coverage

- Write pytest coverage around `CodingMemoryService` that instantiates the service with a temporary project root,
  records turns using synthetic warnings/static findings, and asserts that `record_turn`,
  `_extract_tasks`, `_extract_decisions`, and `build_flow_context` capture the expected summaries/tasks/decisions.
- Add regression coverage for heuristics such as `has_prior_patch_failures` and `is_conflicting_request` so future
  refactorings do not break the flow-continuity safeguards that detect repeated patch-only retries or conflicting
  user requests.

## Flow UX & documentation

- Keep `docs/coding-flows.md` and the README "Coding Flows & Debugging" section synchronized when flow schema/commands
  change (especially new checklist fields, transition reasons, or command flags).
- Expand flow troubleshooting notes with examples for diagnosing stale active flows and conflicting-request prompts in
  long chat sessions.

## Tooling polish

- `missing-docstring` follow-up inventory in `src/mana_analyzer/services/`:
  - `analyze_service.py` (`module`, `AnalyzeService`)
  - `dependency_service.py` (`module`, `DependencyService`)
  - `describe_service.py` (`module`, `DescribeService`)
  - `index_service.py` (`module`, `IndexService`)
  - `llm_analyze_service.py` (`module`, `LlmAnalyzeService`)
  - `report_service.py` (`module`, `ReportService`)
  - `search_service.py` (`module`, `SearchService`)
  - `vulnerability_service.py` (`module`)
  - `__init__.py` (`module`)
