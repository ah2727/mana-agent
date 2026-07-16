---
name: skill-creator
description: Trusted built-in capability that converts verified task experience into reviewable skill proposals without activation.
triggers:
  - A substantial completed task has recorded reusable workflow evidence.
required_tools:
  - recorded_task_events
  - proposal_storage
required_permissions:
  - repository_read
risk_level: medium
version: 1.0.0
---

# Purpose

Convert verified, reusable experience into a redacted proposal for explicit user review.

# When to use

- After a substantial completed and verified task passes deterministic eligibility gates.

# When not to use

- For trivial, failed, unverified, secret-bearing, duplicate, or recursive proposal work.

# Preconditions

- Recorded task evidence exists and a validated structured model decision recommends a reusable procedure.

# Procedure

1. Evaluate recorded facts with deterministic minimum gates and confidence weights.
2. Ask the model generator for a typed generalized procedure.
3. Redact evidence, validate structure, permissions, safety, paths, and duplicates.
4. Store the proposal outside active skill roots for explicit review.

# Safety constraints

- Never install, activate, or overwrite an active skill.
- Never trust model-generated evidence in place of recorded task events.
- Never generate proposals about this capability or its validator and installer.

# Verification

- Validate all typed artifacts and ensure the proposal is absent from active skill indexes.

# Failure recovery

- Quarantine malformed or unsafe output and preserve a validation report.

# Evidence provenance

Evidence is drawn only from recorded sessions, tasks, decisions, tools, mutations, and verification results.
