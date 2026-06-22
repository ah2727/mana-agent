# Goal Profiles

Goal profiles keep task-specific candidate discovery out of the run-state layer.
`RunStateStore` may activate a profile, ask it whether a discovered path is
relevant, and use it to rank pending reads. It must not hardcode heuristics for
one goal type.

Profiles live in `src/mana_analyzer/llm/goal_profiles.py`. To add one:

1. Define a `GoalProfile` or subclass with a stable `id`.
2. Implement `goal_matcher` for the user task wording.
3. Provide `discovery_globs` for deterministic file discovery.
4. Add include/exclude/content rules and a `priority_fn` when read order matters.
5. Register it in `BUILTIN_GOAL_PROFILES`.
6. Add focused tests for matching, relevance, priority, and run-state queue behavior.

Examples:

- `ModelDocsGoalProfile`: finds model/schema files and `docs/models.md`.
- API docs profile: route and schema files plus API documentation.
- Admin docs profile: admin classes, permissions, and admin docs.
- Frontend component profile: component files, stories, tests, and component docs.

Goal-specific heuristics belong in profiles because run-state persistence needs
to stay generic and resumable across task types. Adding a new goal should not
require editing `RunStateStore`.
