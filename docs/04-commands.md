# Commands

`mana-agent` is installed as the `mana-agent` console script from `pyproject.toml`, and the CLI entry point is wired to `mana_agent.commands.cli:app`. [pyproject.toml:1-51](pyproject.toml:1-51)

The project exposes these top-level commands in the Typer app:

- `analyze`
- `ask`
- `chat`
- `continue`

The first three are documented in the README and implemented as Typer commands in the CLI modules. The `continue` command is registered in `src/mana_agent/commands/cli_internal.py`. [README.md:1-242](README.md:1-242) [src/mana_agent/commands/analyze_cli.py:220-360](src/mana_agent/commands/analyze_cli.py:220-360) [src/mana_agent/commands/ask_cli.py:1-262](src/mana_agent/commands/ask_cli.py:1-262) [src/mana_agent/commands/chat_cli.py:120-260](src/mana_agent/commands/chat_cli.py:120-260) [src/mana_agent/commands/cli_internal.py:1-220](src/mana_agent/commands/cli_internal.py:1-220)

## Global usage

The README shows the general pattern for invoking the CLI and a few global flags:

```bash
mana-agent analyze /path/to/project
mana-agent --verbose analyze .
mana-agent --log-dir .mana/logs ask "summarize the parser"
mana-agent --output-dir .mana/output chat
```

`mana-agent --help` is also available, and the README states that `analyze`, `ask`, and `chat` support `--json` where structured output is available. [README.md:1-242](README.md:1-242)

## `analyze`

`analyze` is the unified repository-analysis pipeline. It takes a target path, resolves the project root, indexes the repository, runs dependency analysis, static analysis, semantic search when `--query` is supplied, LLM-assisted findings, structure analysis, and vulnerability scanning, then writes report artifacts under the analyzed project’s `.mana/` directory. [src/mana_agent/commands/analyze_cli.py:220-360](src/mana_agent/commands/analyze_cli.py:220-360) [README.md:1-242](README.md:1-242)

Example:

```bash
mana-agent analyze /path/to/project --query "authentication flow"
mana-agent analyze /path/to/project --json
```

Common options documented in the README and visible in the command signature include:

- `--query`
- `--k`
- `--model`
- `--include-tests/--no-include-tests`
- `--online/--offline`
- `--osv-timeout-seconds`
- `--security-scope`
- `--report-profile`
- `--detail-line-target`
- `--security-lens`
- `--output-format`
- `--fail-on`
- `--auto-continue/--no-auto-continue`
- `--max-passes`
- `--max-tool-calls`
- `--max-runtime-minutes`
- `--max-cost`
- `--json` [src/mana_agent/commands/analyze_cli.py:220-360](src/mana_agent/commands/analyze_cli.py:220-360) [README.md:1-242](README.md:1-242)

Artifacts written by `analyze`:

- `.mana/analyze.json`
- `.mana/analyze.md`
- `.mana/analyze.html`
- `.mana/analyze.dot`
- `.mana/analyze.graphml` [README.md:1-242](README.md:1-242)

## `ask`

`ask` answers a repository question against an index. It supports a direct index, directory-aware mode, ephemeral indexes, agent tool use, and JSON output. When tool mode is enabled, it can return source references and tool trace information. [src/mana_agent/commands/ask_cli.py:1-262](src/mana_agent/commands/ask_cli.py:1-262) [README.md:1-242](README.md:1-242)

Example:

```bash
mana-agent ask "How is configuration loaded?" --root-dir /path/to/project
```

Common options include:

- `--k`
- `--model`
- `--index-dir`
- `--ephemeral-index`
- `--dir-mode`
- `--root-dir`
- `--max-indexes`
- `--auto-index-missing/--no-auto-index-missing`
- `--agent-tools/--no-agent-tools`
- `--agent-max-steps`
- `--agent-unlimited/--no-agent-unlimited`
- `--agent-timeout-seconds`
- `--json` [src/mana_agent/commands/ask_cli.py:1-262](src/mana_agent/commands/ask_cli.py:1-262)

## `chat`

`chat` opens an interactive REPL for repository analysis and coding-agent workflows. Its signature shows support for index selection, directory-aware mode, ephemeral indexes, coding memory, flow IDs, planning mode, auto execution, tool-worker execution, Redis-backed tool execution, and several limits for search, read, and execution budgets. [src/mana_agent/commands/chat_cli.py:120-260](src/mana_agent/commands/chat_cli.py:120-260) [src/mana_agent/commands/cli_internal.py:1-220](src/mana_agent/commands/cli_internal.py:1-220)

Example:

```bash
mana-agent chat --root-dir /path/to/project
```

Notable options visible in the command definition include:

- `--model`
- `--index-dir`
- `--k`
- `--ephemeral-index`
- `--dir-mode`
- `--root-dir`
- `--max-indexes`
- `--auto-index-missing/--no-auto-index-missing`
- `--agent-tools/--no-agent-tools`
- `--coding-agent/--no-coding-agent`
- `--tool-worker-process/--no-tool-worker-process`
- `--tool-worker-strict/--no-tool-worker-strict`
- `--tool-exec-backend`
- `--redis-url`
- `--toolsmanager-parallel-requests`
- `--redis-queue-name`
- `--redis-ttl-seconds`
- `--coding-memory/--no-coding-memory`
- `--flow-id`
- `--coding-plan-max-steps`
- `--coding-search-budget`
- `--coding-read-budget`
- `--coding-require-read-files`
- `--planning-mode`
- `--planning-max-questions`
- `--auto-execute-plan/--no-auto-execute-plan`
- `--auto-execute-max-passes`
- `--auto-continue/--no-auto-continue` [src/mana_agent/commands/chat_cli.py:120-260](src/mana_agent/commands/chat_cli.py:120-260)

The README also notes that chat can persist coding memory at `<project>/.mana/index/chat_memory.sqlite3`. [README.md:1-242](README.md:1-242)

### In-chat slash command: `/analyze`

Inside the chat REPL you can run `/analyze` to analyze the current project and write report artifacts under `.mana/`. It is read-only except for those artifacts and is handled before the message is sent to the LLM. [src/mana_agent/commands/chat_analyze_command.py:1-80](src/mana_agent/commands/chat_analyze_command.py:1-80) [src/mana_agent/commands/chat_cli.py:1418-1470](src/mana_agent/commands/chat_cli.py:1418-1470)

Typing `/analyze` with no arguments opens a format menu:

```text
Select output format:

1. JSON
2. Markdown
3. HTML
4. DOT graph
5. GraphML
6. Mermaid diagram
7. All formats

Enter choice:
```

Enter a single number (`1`), several numbers (`1,2,3`), or `7` for all formats. You can also skip the menu with direct forms:

```text
/analyze all
/analyze json
/analyze markdown
/analyze md
/analyze html
/analyze dot
/analyze graphml
/analyze mermaid
/analyze json markdown html
/analyze --format json,markdown,html
```

Format-to-artifact mapping (written under `.mana/`):

| Token | Artifact |
| --- | --- |
| `json` | `.mana/analyze.json` |
| `markdown` / `md` | `.mana/analyze.md` |
| `html` | `.mana/analyze.html` |
| `dot` | `.mana/analyze.dot` |
| `graphml` | `.mana/analyze.graphml` |
| `mermaid` | `.mana/diagram.mmd` |
| `all` | every artifact above |

Invalid tokens (e.g. `/analyze pdf`) print a clean error listing the supported formats and do not crash the session. The canonical format list lives in [src/mana_agent/commands/analyze_formats.py:1-80](src/mana_agent/commands/analyze_formats.py:1-80).

## `continue`

`continue` resumes a persisted auto-execute run from `<root>/.mana/runs/<run_id>`. It requires `--run-id` and accepts caps for passes, tool calls, runtime, and cost, plus retrieval and step limits. [src/mana_agent/commands/cli_internal.py:1-220](src/mana_agent/commands/cli_internal.py:1-220)

Common options visible in the command definition include:

- `--run-id`
- `--root-dir`
- `--pass-cap`
- `--auto-continue/--no-auto-continue`
- `--max-passes`
- `--max-tool-calls/--max-total-tool-calls`
- `--max-runtime-minutes`
- `--max-cost`
- `--max-no-progress-passes`
- `--timeout`
- `--k`
- `--max-steps`
- `--max-resume-cycles` [src/mana_agent/commands/cli_internal.py:1-220](src/mana_agent/commands/cli_internal.py:1-220)

## Help and verification

All commands support `--help`. The README also lists local verification commands that are useful after editing CLI docs or behavior:

```bash
pytest -q
ruff check src tests
mypy src tests
python -c "import mana_agent; print('ok')"
mana-agent --help
mana-agent analyze --help
mana-agent ask --help
mana-agent chat --help
```

[README.md:1-242](README.md:1-242)
