# Agent Behavior

This document describes the expected behavior of the coding agent used by
`mana-agent`.

## Behavior Principles

The agent should:

- understand the user request and current context,
- gather repository evidence before concluding,
- prefer direct citations from repository files,
- make concrete changes only when evidence supports them,
- run checks after edits when possible,
- report what changed and what was not verified.

## Typical Workflow

1. Clarify the task.
2. Search the repository for relevant code or docs.
3. Read the source files that support the answer or change.
4. Edit only the necessary files.
5. Verify the change with tests or smoke checks.
6. Summarize the result with file citations.

## Reporting Expectations

When finishing a task, the agent should report:

- changed files,
- key checks run,
- any skipped checks,
- remaining risks or unknowns.

## In-chat Slash Commands

Some chat inputs are intercepted before the model runs. These are deterministic,
read-only operations that never invoke the LLM or coding agent:

- `/flow` — inspect or reset the active coding flow.
- `/analyze` — analyze the current project and write report artifacts under
  `.mana/` (`json`, `markdown`/`md`, `html`, `dot`, `graphml`, `mermaid`, or
  `all`). With no arguments it opens a numbered format menu. The only side effect
  is writing the selected `.mana/` artifacts; source files are never modified.

Anything that is not a recognized slash command is treated as a normal request
and routed to the agent as usual.

## Related Docs

- [Architecture](./08-architecture.md)
- [Tool System](./13-tool-system.md)
- [README](../README.md)
