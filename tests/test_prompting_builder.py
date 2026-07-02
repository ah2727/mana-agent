from pathlib import Path

from mana_agent.agent.flow import FLOW_ORDER, build_agent_flow
from mana_agent.agent.selection import AgentPhase
from mana_agent.prompting.builder import build_coding_system_prompt
from mana_agent.prompting.layers import PROMPT_LAYER_ORDER, PromptLayer, compose_layers


def test_build_agent_flow_connects_selection_context_and_verification(tmp_path: Path) -> None:
    flow = build_agent_flow(
        "Fix prompt builder flow in src/mana_agent/llm/coding_agent.py",
        repo_root=tmp_path,
        candidate_files=("src/mana_agent/llm/coding_agent.py",),
    )

    assert FLOW_ORDER[0] is AgentPhase.DISCOVER
    assert flow.context.mode == "edit"
    assert flow.context.phase is AgentPhase.READ
    assert flow.context.repo_root == tmp_path.resolve()
    assert "src/mana_agent/llm/coding_agent.py" in flow.context.candidate_files
    assert "src/mana_agent/llm/coding_agent.py" in flow.context.candidate_search_terms
    assert flow.context.done_criteria
    assert flow.verification.commands or flow.verification.notes


def test_coding_prompt_builder_composes_stable_layers(tmp_path: Path) -> None:
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills" / "testing.md").write_text("# Testing Skill\n\nUse focused checks.\n", encoding="utf-8")
    (tmp_path / ".mana").mkdir()
    (tmp_path / ".mana" / "memory.md").write_text("Known command: pytest -q\n", encoding="utf-8")

    prompt = build_coding_system_prompt(
        base_prompt="Core Identity",
        request="Add pytest coverage for prompt builder",
        repo_root=tmp_path,
        full_auto_mode=True,
        include_edit_rules=True,
        flow_context="Flow ID: abc123",
    )

    assert prompt.index("Core Identity") < prompt.index("Language-aware tooling")
    assert prompt.index("Mode Rules") < prompt.index("Compact Skills Index")
    assert prompt.index("Compact Skills Index") < prompt.index("Project Memory Snapshot")
    assert prompt.index("Project Memory Snapshot") < prompt.index("Current Task Context")
    assert prompt.index("Current Task Context") < prompt.index("Output Contract")
    assert "testing (project): Testing Skill" in prompt
    assert "Project Memory Snapshot" in prompt
    assert "Known command: pytest -q" in prompt
    assert "Current Task Context" in prompt
    assert "explicit_requirements" in prompt
    assert "done_criteria" in prompt
    assert "Output Contract" in prompt
    assert "Flow ID: abc123" in prompt


def test_compose_layers_rejects_unstable_order() -> None:
    layers = [PromptLayer(name, name) for name in reversed(PROMPT_LAYER_ORDER)]

    try:
        compose_layers(layers)
    except ValueError as exc:
        assert "stable order" in str(exc)
    else:
        raise AssertionError("compose_layers should reject prompts outside the stable order")
