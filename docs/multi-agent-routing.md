# Multi-Agent Routing

Mana Agent routes every LLM-facing request through `mana_agent.multi_agent.MainAgent`.
The old command names remain public, but the internal record starts with a
TaskBoard item, a route decision, agent assignments, and a final SummarizerAgent
summary.

## Hierarchy

```text
MainAgent
  └── HeadDecisionAgent
        ├── PlannerAgent
        ├── ResearchAgent
        ├── CodingAgent
        │     └── CodingSubAgent(s)
        ├── ToolAgent
        ├── VerifierAgent
        ├── ReviewerAgent
        └── SummarizerAgent
```

## TaskBoard

TaskBoard state is persisted in `.mana/taskboard/state.json`; append-only events
are written to `.mana/taskboard/history.jsonl`. Tasks store status, risk,
assigned agents, required capabilities, files, queue jobs, plan, evidence,
assumptions, blockers, discussions, decisions, and verification results.

## Communication And Decisions

Agents exchange concise structured messages through `MessageBus`. Complex,
mutation, ambiguous, or higher-risk requests open a `DecisionRoom`, where
HeadDecisionAgent records the selected route, rationale summary, risks,
assumptions, rejected options, assigned agents, and verification needs.

## Queue And Tools

CodingAgent never executes tools directly. It creates QueueManager jobs.
QueueManager schedules jobs FIFO with priority ordering, serializes write jobs
with locks, and delegates execution to ToolsManager. ToolsManager wraps the
existing repository-safe commands and blocks dangerous shell operations such as
`rm -rf /`, `.env` reads, `printenv`, `git reset --hard`, and `git clean -fd`.

## Verification

VerifierAgent records verification requirements for every mutation route and
stores `VerificationResult` rows on the TaskBoard. Existing command paths still
run their concrete tests or analyze flows after the mandatory multi-agent route
has been recorded.

## CLI Behavior

- `mana-agent chat` records each user turn through MainAgent.
- `/analyze` inside chat records an analyze route before running the analyzer.
- `/plan` inside chat records a planning route before generating a plan answer.
- `mana-agent analyze` records an analyze route before generating artifacts.
- `mana-agent plan` records a planning route before rendering/saving the plan.
- Coding/edit turns record a coding route with PlannerAgent, CodingAgent,
  QueueManager, ToolAgent, VerifierAgent, ReviewerAgent, and SummarizerAgent.

There is no `--no-multi-agent` flag, `MANA_MULTI_AGENT=0` bypass, or config key
that disables multi-agent routing.
