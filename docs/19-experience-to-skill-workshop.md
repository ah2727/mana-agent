# Experience-to-Skill Workshop

The Experience-to-Skill Workshop converts verified task experience into reviewable skill proposals. It never installs or activates a skill automatically.

## Lifecycle

```text
Task completion
  -> deterministic eligibility and confidence evaluation
  -> recorded evidence collection and redaction
  -> typed skill-creator model decision
  -> structural, safety, permission, path, and duplicate validation
  -> ~/.mana/skill-proposals/<proposal-id>/
  -> explicit user review: install, edit, reject, or quarantine
```

Proposal generation runs after the normal task has been acted on, verified, summarized, and recorded. A workshop failure emits a diagnostic event but does not change the original task result. Learning is disabled recursively for `skill_creator`, `proposal_validator`, and `proposal_installer` work.

## Eligibility and confidence

Code-level gates require a completed multi-step task with a meaningful mutation or artifact, the configured number of successful runs, and successful verification when required. Failed, abandoned, unresolved, highly repository-specific, recursive, and trivial work is rejected before a model is asked to generate anything.

| Signal | Weight |
| --- | ---: |
| Successful verification | +0.25 |
| Explicit user acceptance | +0.20 |
| More than one successful run | +0.15 |
| Clear reusable trigger | +0.10 |
| Deterministic verification | +0.10 |
| Low repository specificity | +0.10 |
| No unresolved warnings | +0.10 |

Penalties are applied for unresolved corrections, failed or partial verification, high repository specificity, missing safety constraints, and incomplete evidence. Scores are clamped to `0.0..1.0`. By default, `0.80+` creates `pending_review`, `0.60..0.79` creates `needs_attention`, and lower-scored experience is not stored as a normal proposal.

## Storage and trust boundaries

```text
~/.mana/
├── skills/             # explicitly approved active skills
├── skill-proposals/    # untrusted proposals awaiting review
└── skill-quarantine/   # unsafe, malformed, suspicious, or manually quarantined proposals
```

Each proposal contains `proposal.yaml`, `SKILL.md`, `evidence.json`, `validation.json`, and `README.md`. `proposal.yaml` is YAML-compatible JSON, allowing the existing JSON serializer to be reused without a new dependency. Typed Pydantic models validate proposal data before writes. Temporary files, atomic replacement, stable IDs, and a cross-process lock protect creation and lifecycle transitions.

Pending and quarantined directories are never searched by `SkillManager` or injected into prompts. Installation copies a revalidated `SKILL.md` into the active root and writes `provenance.json` with version, source sessions/tasks, installation time, proposal ID, and evidence hash. Existing active names are never overwritten; upgrades require a separate explicit flow.

## Evidence, duplicates, and security

Evidence is reconstructed from recorded sessions, taskboard records, decisions, tool results, changed-file references, verification results, corrections, agent IDs, and Git metadata. The model generator cannot provide replacement evidence.

Nested secret-like fields and text are redacted before persistence. Validation rejects unsupported permissions, traversal names, missing sections, missing verification evidence, likely secrets, personal absolute paths, and destructive instructions without an explicit approval constraint.

Duplicate comparison covers active skills, proposals, quarantined entries, aliases, descriptions, triggers, and procedures. Duplicate proposal evidence is merged under a lock. Active skills remain immutable and receive separate append-only supporting evidence.

## CLI and review actions

```bash
mana-agent skill proposals
mana-agent skill proposals --status pending_review --min-confidence 0.8 --risk medium
mana-agent skill proposal show <proposal-id>
mana-agent skill proposal review <proposal-id>
mana-agent skill proposal edit <proposal-id> --draft-file edited-draft.json
mana-agent skill proposal edit <proposal-id> --skill-file edited-SKILL.md
mana-agent skill proposal install <proposal-id> --version 1.0.0
mana-agent skill proposal reject <proposal-id> --reason "too broad"
mana-agent skill proposal quarantine <proposal-id> --reason "unsafe command"
mana-agent skill create-from-session <session-id>
```

`create-from-session` loads recorded session memory and uses the automatic pipeline. If no model is configured, it stops with an explicit decision error and performs no fallback generation. A pre-generated typed `SkillDraft` JSON may be supplied with `--draft-file` for offline review.

To recover a quarantined proposal, inspect its evidence and report, correct the source issue outside the active root, and run a fresh validated proposal workflow. Quarantined proposals cannot be installed directly.

## Configuration

In `~/.mana/config.toml`:

```toml
[experience_to_skill]
enabled = true
auto_propose = true
minimum_confidence = 0.80
needs_attention_confidence = 0.60
minimum_successful_runs = 1
require_verification = true
require_user_acceptance = false
semantic_duplicate_threshold = 0.88
retain_rejected_days = 90
quarantine_on_validation_failure = true
```

Environment overrides:

```text
MANA_EXPERIENCE_TO_SKILL_ENABLED
MANA_EXPERIENCE_TO_SKILL_AUTO_PROPOSE
MANA_EXPERIENCE_TO_SKILL_MINIMUM_CONFIDENCE
MANA_EXPERIENCE_TO_SKILL_NEEDS_ATTENTION_CONFIDENCE
MANA_EXPERIENCE_TO_SKILL_MINIMUM_SUCCESSFUL_RUNS
MANA_EXPERIENCE_TO_SKILL_REQUIRE_VERIFICATION
MANA_EXPERIENCE_TO_SKILL_REQUIRE_USER_ACCEPTANCE
MANA_EXPERIENCE_TO_SKILL_DUPLICATE_THRESHOLD
MANA_EXPERIENCE_TO_SKILL_RETAIN_REJECTED_DAYS
MANA_EXPERIENCE_TO_SKILL_QUARANTINE_ON_FAILURE
MANA_SKILLS_ROOT
MANA_SKILL_PROPOSALS_ROOT
MANA_SKILL_QUARANTINE_ROOT
```

## Events and dashboard consumers

The shared execution event hub publishes `skill_candidate_detected`, `skill_proposal_generation_started`, `skill_proposal_created`, `skill_proposal_validation_failed`, `skill_proposal_quarantined`, `skill_proposal_installed`, and `skill_proposal_rejected`. Dashboard/API consumers can build filtered lists, evidence views, editors, warnings, actions, and installed-version history from the same storage and event contracts; no dashboard-specific lifecycle implementation is required.
