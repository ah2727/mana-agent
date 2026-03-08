# Agent Instructions

## Mutation Rules

Use repository mutation tools directly. Do not use `git diff` / `git apply` workflows for code edits.

### Preferred edit flow

1. Validate intent and target files.
2. Apply edits with `apply_patch`.
3. If patch attempts fail or no-op, switch to `write_file` fallback immediately.
4. Verify real file changes.
5. Stop with a clear blocker summary if both mutation paths fail.

### Required constraints

- Never rely on `git diff`, `git apply`, or `git format-patch` as the edit path.
- Keep edits inside repo path constraints and allowed prefixes.
- Preserve anti-loop behavior: no repeated patch-only loops.
- Prefer minimal, scoped changes and explicit verification.

### Anti-loop requirement

If patch attempts fail or no files change:

1. Do not repeat a patch-only strategy in a loop.
2. Switch to a non-patch fallback (`write_file`) immediately.
3. If fallback also produces no file changes, stop and return a clear failure summary
   (what was attempted, why it likely failed, and what file/path constraints blocked progress).
