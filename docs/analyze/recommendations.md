# Recommendations

## LLM Recommendations
- Clarify dependency lock policy by documenting the intended pip workflow between pyproject.toml and requirements.txt, and ensure CI consistently installs from the chosen files (evidence: dependency warning about missing poetry.lock).
- Add/verify timeouts around any long-running subprocess/command execution helpers used by CLI modes and agent work queue paths (evidence: recommendation mentions specific files, but no direct timeout evidence was detected).
- Ensure agent context artifacts (e.g., .mana/analyze/agent_context.json) are loaded for chat context grounding and never include secrets (evidence: recommendation states this artifact exists, but the artifact path is not detected in the file listing; treat as not detected and verify before implementing).

## Next Coding Tasks (LLM)
### Block/guard destructive repository operations with explicit confirmation and safe targeting
- Priority: High
- Files: `src/mana_agent/tools/repository.py`, `src/mana_agent/llm/ask_agent.py`
- Acceptance criteria:
  - No rm -rf or git reset --hard commands execute without explicit, user-confirmed intent and strict path validation (evidence-based scope tied to the detected command strings).
  - Existing tests continue to pass, and new tests (if added) assert that unsafe commands are rejected/blocked.
- Verification: `pytest -q`

### Remove any committed secrets from tracked files and ensure tests/docs never write real credentials
- Priority: High
- Files: `README.md`, `CHANGELOG.md`, `docs/03-quick-start.md`, `tests/test_chat_planning_mode.py`, `tests/test_cli_smoke.py`, `tests/test_ask_agent_recovery.py`, `tests/test_tool_worker_process.py`
- Acceptance criteria:
  - Replace secret-bearing values with environment-variable references and ensure .env.example contains only non-secret placeholders.
  - Add/enable a repository-level check that fails if secret patterns (e.g., OPENAI_/openai_) appear in tracked files.
- Verification: `python -m compileall . && pytest -q`

### Stabilize test suite and CI by aligning dependency installation policy
- Priority: Medium
- Files: `pyproject.toml`, `requirements.txt`
- Acceptance criteria:
  - Document whether pip installs from requirements.txt or from pyproject.toml and how version pinning is enforced.
  - CI installs deterministically and compilation succeeds.
- Verification: `python -m compileall src`

## Deterministic Recommendations

### Clarify dependency lock policy
- Priority: medium
- Reason: Dependency manifests have lock-file warnings.
- Verification: `python -m compileall src`
- Acceptance criteria:
  - Document the intended package manager and lock workflow.
  - CI verifies dependency installation from the chosen files.

### Add timeout around long-running verification commands
- Priority: high
- Reason: Subprocess usage without nearby timeout evidence can stall agent runs.
- Verification: `pytest tests -q`
- Acceptance criteria:
  - All command execution helpers accept and enforce a timeout.
  - Timeout behavior has a focused test.

### Load agent_context.json into future chat context
- Priority: medium
- Reason: The compact context artifact is designed to make later coding-agent turns faster and better grounded.
- Verification: `pytest tests/commands tests/integration -q`
- Acceptance criteria:
  - Chat startup checks for .mana/analyze/agent_context.json.
  - Loaded context is bounded and never includes secrets.
