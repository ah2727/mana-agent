# Codex coding backend

Mana-Agent can host Codex as an optional coding executor while retaining control
of intent decisions, task decomposition, worktree allocation, permissions,
verification, memory, and final reporting.

The official Codex SDK is currently TypeScript (`@openai/codex-sdk`) and wraps
the Codex CLI. There is no official `openai-codex` Python package or
`AsyncCodex` client. Mana-Agent therefore integrates from Python through the
official `codex app-server` JSON-RPC protocol, which exposes thread and turn
lifecycle methods, streaming notifications, and `turn/interrupt` cancellation.
See the [official Codex SDK package](https://www.npmjs.com/package/@openai/codex-sdk)
and [Codex app-server protocol](https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md).

## Responsibility boundary

```text
Mana model decision
  → validated coding backend decision
    → Mana-managed isolated worktree
      → Codex thread and turn
        → normalized events and result
          → independent Mana verification and review
```

Codex never selects itself. `CodingBackendDecision` is a typed model decision
that names exactly one registered backend. Missing, invalid, unsafe, disabled,
or unavailable selections stop with an explicit error. Mana-Agent does not
silently choose the native backend.

Writing tasks require a separate clean Git worktree. The Codex prompt prohibits
commits, pushes, publishing, credential access, and permission elevation.
Approval requests become failed task results for Mana-Agent to surface; the
backend never self-approves them.

## Installation and authentication

Install the official Codex CLI using an OpenAI-supported installation method,
then authenticate it:

```bash
codex login
mana-agent codex status --repo .
mana-agent codex doctor --repo .
```

`mana-agent codex login` and `mana-agent codex logout` delegate directly to the
official CLI. Mana-Agent does not read or copy Codex credentials.

## Configuration

Add these values to `~/.mana/config.toml`:

```toml
MANA_CODEX_ENABLED = true
MANA_CODEX_BIN = "codex"
MANA_CODEX_MAX_WORKERS = 2
MANA_CODEX_STREAM_EVENTS = true
MANA_CODEX_WORKTREE_ISOLATION = true
MANA_CODEX_TASK_TIMEOUT_SECONDS = 1800
MANA_CODEX_ALLOW_NETWORK = false
MANA_CODEX_MODEL = ""
```

Codex is disabled by default. Network access remains disabled by policy unless
a future validated execution decision and sandbox implementation explicitly
support it.

## Runtime contracts

- `mana_agent.coding` contains provider-neutral task, workspace, result, event,
  registry, and orchestrator contracts.
- `mana_agent.integrations.codex` owns the app-server process, protocol,
  prompts, event mapping, result parsing, health checks, and backend.
- `CodexWorkerPool` bounds concurrency and serializes tasks whose declared file
  scopes overlap. Empty scopes are treated conservatively as overlapping.
- Each logical coding task starts one Codex thread. Repair turns may reuse that
  thread when the caller retains its thread ID; unrelated tasks must not.

Codex-reported tests are evidence only. The returned `CodingTaskResult` must be
passed to Mana-Agent's independent verifier before integration or completion.
