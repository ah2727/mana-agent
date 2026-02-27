import json
from pathlib import Path

from typer.testing import CliRunner

from mana_analyzer.commands.cli import app
from mana_analyzer.services.coding_memory_service import CodingMemoryService

runner = CliRunner()


class DummySettings:
    openai_api_key = "test"
    openai_base_url = None
    openai_chat_model = "fake"
    openai_embed_model = "fake"
    default_top_k = 8
    coding_flow_max_turns = 5
    coding_flow_max_tasks = 20
    coding_plan_max_steps = 8
    coding_search_budget = 4
    coding_read_budget = 6
    coding_require_read_files = 2


class _AskServiceWithAgent:
    def __init__(self) -> None:
        class _AskAgent:
            def __init__(self) -> None:
                self.tools: list[object] = []
                self.model = "fake"

            def ask(self, question: str, **kwargs: object) -> str:
                _ = (question, kwargs)
                return "ok"

        self.ask_agent = _AskAgent()


def _seed_flow(project_root: Path) -> str:
    service = CodingMemoryService(project_root=project_root, max_turns=5, max_tasks=20)
    flow_id = service.ensure_flow(flow_id=None, request="Implement flow summary command and docs")
    service.record_turn(
        flow_id=flow_id,
        user_request="Implement flow summary command and docs",
        effective_prompt="system prompt",
        agent_answer=(
            "Decision: Keep persistence schema unchanged\n"
            "- [x] Wire flow command\n"
            "- [ ] Write docs\n"
        ),
        changed_files=["src/mana_analyzer/commands/cli.py", "README.md"],
        warnings=["write_file fallback was used once"],
        static_findings=["missing-docstring: src/mana_analyzer/services/index_service.py:27"],
        checklist={
            "objective": "Implement flow visibility and docs",
            "steps": [
                {"status": "done", "title": "Wire flow command"},
                {"status": "blocked", "title": "Finish docs"},
            ],
        },
        transitions=[
            {"from_phase": "discover", "to_phase": "edit", "reason": "files identified"},
            {"from_phase": "edit", "to_phase": "blocked", "reason": "waiting for docs update"},
        ],
    )
    return flow_id


def test_flow_command_no_active_flow(tmp_path: Path) -> None:
    result = runner.invoke(app, ["flow", str(tmp_path)])
    assert result.exit_code == 0
    assert "No active coding flow found." in result.stdout


def test_flow_command_text_output_includes_core_sections(tmp_path: Path) -> None:
    _seed_flow(tmp_path)
    result = runner.invoke(app, ["flow", str(tmp_path)])
    assert result.exit_code == 0
    assert "Flow" in result.stdout
    assert "Objective" in result.stdout
    assert "Open tasks" in result.stdout
    assert "Flow Checklist" in result.stdout
    assert "Unresolved static findings" in result.stdout
    assert "Last blocked reason" in result.stdout


def test_flow_command_json_output_with_explicit_flow_id(tmp_path: Path) -> None:
    flow_id = _seed_flow(tmp_path)
    result = runner.invoke(app, ["flow", str(tmp_path), "--flow-id", flow_id, "--format", "json"])
    assert result.exit_code == 0
    start = result.stdout.find("{")
    assert start >= 0
    payload = json.loads(result.stdout[start:])
    assert payload["flow_id"] == flow_id
    assert isinstance(payload.get("checklist"), dict)
    assert isinstance(payload.get("open_tasks"), list)
    assert isinstance(payload.get("unresolved_static_findings"), list)


def test_flow_command_includes_blocked_transition_reason(tmp_path: Path) -> None:
    _seed_flow(tmp_path)
    result = runner.invoke(app, ["flow", str(tmp_path)])
    assert result.exit_code == 0
    assert "waiting for docs update" in result.stdout


def test_chat_startup_with_coding_memory_and_coding_agent_still_works(monkeypatch) -> None:
    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr(
        "mana_analyzer.commands.cli.build_ask_service",
        lambda _s, model_override=None: _AskServiceWithAgent(),
    )
    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent", "--coding-memory"],
        input="quit\n",
    )
    assert result.exit_code == 0
    assert "Goodbye!" in result.stdout
