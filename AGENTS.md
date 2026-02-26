# Agent Instructions

## Git Diff / Patch Rules

Use this workflow for creating and applying git patches.

### Create patch files with `git diff`

- Unstaged changes:
  - `git diff > unstaged-changes.patch`
- Staged changes:
  - `git diff --cached > staged-changes.patch`
- Staged + unstaged:
  - `git diff HEAD > all-changes.patch`
- Between two commits:
  - `git diff <commit1> <commit2> > between-commits.patch`
- Between branches:
  - `git diff main..feature-branch > branch-diff.patch`
- Single path:
  - `git diff -- path/to/file > file-changes.patch`

If commit metadata must be preserved (author/date/message), use
`git format-patch` and apply with `git am` instead of `git apply`.

### Apply patch files with `git apply`

Always run this sequence:

1. Dry run:
   - `git apply --check patch_file.patch`
2. Preview scope:
   - `git apply --stat patch_file.patch`
3. Apply:
   - `git apply patch_file.patch`
4. Review:
   - `git diff`
5. Commit:
   - `git add -A && git commit -m "Apply patch"`

### Common `git apply` options

- `--3way` for three-way merge attempts when context drifted.
- `--reject` to apply partial hunks and leave `.rej` files.
- `--whitespace=fix` to auto-fix whitespace issues.
- `-R` / `--reverse` to undo an already applied patch.
- `--directory=<dir>` when patch paths are rooted differently.
- `--exclude=<pattern>` / `--include=<pattern>` for selective apply.
- `--verbose` for per-file diagnostics.
- `--fuzz=<n>` for looser context matching.

### `git apply` vs `git am`

- Use `git apply` for `git diff` patches (working tree changes, no commit metadata).
- Use `git am` for `git format-patch` output (preserves commit metadata).
- Rule of thumb:
  - Starts with `From <hash>` => `git am`
  - Starts with `diff --git` => `git apply`

### Anti-loop requirement

If patch attempts fail or no files change:

1. Do not repeat a patch-only strategy in a loop.
2. Switch to a non-patch fallback (`write_file`) immediately.
3. If fallback also produces no file changes, stop and return a clear failure summary
   (what was attempted, why it likely failed, and what file/path constraints blocked progress).
