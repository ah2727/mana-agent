from __future__ import annotations

from pathlib import Path

import pytest

from mana_agent.skills.adaptive import (
    RepositoryIdentityService,
    SkillCandidateGenerator,
    SkillEvidence,
    SkillManifest,
    SkillSelector,
    SkillStorage,
    skills_root,
)
from mana_agent.config.skills import AdaptiveSkillsConfig
from mana_agent.skills.chat import ChatSkillCoordinator


def _candidate(repository_id: str) -> tuple[str, SkillManifest, SkillEvidence]:
    manifest = SkillManifest(
        id="skill_safe_test_01", name="safe-test", title="Safe test", description="A verified procedure.",
        repository={"id": repository_id}, applicability={"required_tools": ["repo_read"]},
    )
    evidence = SkillEvidence(repository_id=repository_id, verification_results=[{"command": "pytest", "passed": True}])
    markdown = """# Safe test

## Use when

The planner selected this verified procedure.

## Do not use when

The required tool is unavailable.

## Required context

Inspect the selected repository files.

## Procedure

1. Follow the evidence-backed steps.

## Verification

Run the recorded verification.

## Failure recovery

Stop and report the failed verification.
"""
    return markdown, manifest, evidence


def test_mana_home_resolves_adaptive_skill_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    assert skills_root() == (tmp_path / "mana" / "skills").resolve()


def test_candidate_stays_outside_repository_until_explicit_approval(tmp_path: Path) -> None:
    repo = tmp_path / "repo"; repo.mkdir()
    identity = RepositoryIdentityService(tmp_path / "mana").identify(repo)
    storage = SkillStorage(tmp_path / "mana")
    markdown, manifest, evidence = _candidate(identity.repository_id)
    candidate = SkillCandidateGenerator(storage).create(markdown=markdown, manifest=manifest, evidence=evidence)
    assert candidate.is_relative_to(tmp_path / "mana" / "skills")
    assert not (repo / ".mana" / "skills").exists()
    active = storage.activate(identity.repository_id, manifest.id)
    assert (active / "versions" / "1.0.0" / "manifest.yaml").exists()


def test_selection_requires_explicit_decision_and_available_tools(tmp_path: Path) -> None:
    repo = tmp_path / "repo"; repo.mkdir(); identity = RepositoryIdentityService(tmp_path / "mana").identify(repo)
    storage = SkillStorage(tmp_path / "mana"); markdown, manifest, evidence = _candidate(identity.repository_id)
    SkillCandidateGenerator(storage).create(markdown=markdown, manifest=manifest, evidence=evidence); storage.activate(identity.repository_id, manifest.id)
    decisions, selected = SkillSelector(storage).select(identity.repository_id, [manifest.id], available_tools=[])
    assert not selected and not decisions[0].selected
    decisions, selected = SkillSelector(storage).select(identity.repository_id, [manifest.id], available_tools=["repo_read"])
    assert [item.id for item in selected] == [manifest.id] and decisions[0].selected


def test_chat_context_is_repository_isolated_and_loads_only_selected_skills(tmp_path: Path) -> None:
    mana = tmp_path / "mana"
    first = tmp_path / "first"; first.mkdir()
    second = tmp_path / "second"; second.mkdir()
    identities = RepositoryIdentityService(mana)
    first_id = identities.identify(first).repository_id
    storage = SkillStorage(mana)
    markdown, manifest, evidence = _candidate(first_id)
    SkillCandidateGenerator(storage).create(markdown=markdown, manifest=manifest, evidence=evidence)
    storage.activate(first_id, manifest.id)
    coordinator = ChatSkillCoordinator(config=AdaptiveSkillsConfig(root_path=mana), storage=storage)
    one = coordinator.initialize_session(first)
    two = coordinator.initialize_session(second)
    assert [item.id for item in one.available_skills] == [manifest.id]
    assert not two.available_skills
    assert not coordinator.load_selected(one)
    decisions = coordinator.select_for_task(one, selected_ids=[manifest.id], available_tools=["repo_read"])
    assert decisions[0].selected
    assert "## Adaptive skill: safe-test v1.0.0" in coordinator.load_selected(one)


def test_chat_session_disable_clears_selected_skills(tmp_path: Path) -> None:
    coordinator = ChatSkillCoordinator(config=AdaptiveSkillsConfig(root_path=tmp_path / "mana"))
    context = coordinator.initialize_session(tmp_path)
    assert "disabled for this session" in coordinator.render_command(context, "/skills disable")
    assert not context.enabled
    assert "enabled for this session" in coordinator.render_command(context, "/skills enable")
