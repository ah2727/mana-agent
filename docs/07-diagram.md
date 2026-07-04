# Project Diagram

This page contains Mermaid diagrams that summarize how `mana-agent`
coordinates repository discovery, artifact generation, and coding workflows.

## High-level flow

```mermaid
flowchart TD
    A["Local repository"] --> B["Discover and index files"]
    B --> C["Analyze repository"]
    C --> D["Generate artifacts under .mana"]
    B --> E["Search and retrieve evidence"]
    E --> F["LLM answers grounded in evidence"]
    B --> G["Chat REPL"]
    G --> H["Plan"]
    H --> I["Inspect files and retrieve evidence"]
    I --> J["Patch or write files through tools"]
    J --> K["Run verification"]
    K --> L["Summarize changes"]
```

## Artifact outputs

```mermaid
flowchart LR
    A["Requested formats"] --> B["analyze.json"]
    A --> C["analyze.md"]
    A --> D["analyze.html"]
    A --> E["analyze.dot or graphml"]
    A --> F["diagram.mmd"]

    B --> G["Automation and CI"]
    C --> H["Human review"]
    D --> I["Browseable report"]
    E --> J["Graph visualization"]
    F --> K["Embeddable diagram"]
```

## Coding-agent tool lifecycle

```mermaid
sequenceDiagram
    participant U as User
    participant R as QueueManager / Planner
    participant W as Worker / Tool Runner
    participant T as Repository Tools

    U->>R: Coding request
    R->>T: Search and read files
    T-->>R: Evidence sources

    R->>W: Planned mutation intent
    W->>T: Read, search, write, patch, or delete
    T-->>W: Changed file list

    W->>T: Run verification when available
    T-->>W: Verification results

    W-->>R: Final changed files and summary
    R-->>U: Answer with changed files
```
