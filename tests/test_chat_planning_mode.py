from pathlib import Path

import pytest
from typer.testing import CliRunner

from mana_analyzer.analysis.models import AskResponse, SearchHit
from mana_analyzer.commands import cli

runner = CliRunner()


class DummySettings:
    openai_api_key = "test"
    openai_base_url = None
    openai_chat_model = "fake"
    openai_tool_worker_model = None
    openai_coding_planner_model = None
    openai_embed_model = "fake"
    default_top_k = 8


class RecordingAskService:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def ask(self, index_dir: str, question: str, k: int) -> AskResponse:
        _ = index_dir
        _ = k
        self.calls.append(question)
        return AskResponse(
            answer="Plan response",
            sources=[
                SearchHit(0.9, "/tmp/a.py", 1, 5, "a", "snippet"),
                SearchHit(0.8, "/tmp/b.py", 2, 8, "b", "snippet"),
            ],
        )


def test_chat_planning_mode_asks_questions_and_resets(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr(
        "mana_analyzer.commands.cli.build_ask_service",
        lambda _s, model_override=None: RecordingAskService(calls),
    )

    user_input = "\n".join(
        [
            "plan auth module",
            "success means no open implementation choices",
            "scope src/auth only; no schema changes",
            "markdown with milestones and tests",
            "plan billing module",
            "success means migration-safe rollout",
            "scope billing services and CLI flags",
            "include rollout and rollback checks",
            "quit",
        ]
    ) + "\n"

    result = runner.invoke(
        cli.app,
        ["chat", "--planning-mode", "--planning-max-questions", "3"],
        input=user_input,
    )

    assert result.exit_code == 0
    assert result.stdout.count("Planning question 1/3") == 2
    assert "Generating decision-complete plan" in result.stdout
    assert len(calls) == 2
    assert "You are in planning mode." in calls[0]
    assert "plan auth module" in calls[0]
    assert "A3: markdown with milestones and tests" in calls[0]
    assert "plan billing module" in calls[1]


def test_main_warns_for_python_314(monkeypatch) -> None:
    monkeypatch.setattr("mana_analyzer.commands.cli.setup_logging", lambda **_: Path("/tmp/mana.log"))
    monkeypatch.setattr(cli.sys, "version_info", (3, 14, 0), raising=False)

    with pytest.warns(UserWarning, match="Python 3.14"):
        cli.main(verbose=False, debug_llm=False, log_dir=None, output_dir=None)


def test_chat_planning_mode_uses_llm_generated_questions(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []
    generated_args: list[dict] = []

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr(
        "mana_analyzer.commands.cli.build_ask_service",
        lambda _s, model_override=None: RecordingAskService(calls),
    )

    def _fake_llm_question(
        *,
        ask_service,
        planning_request: str,
        prior_questions: list[str],
        prior_answers: list[str],
        asked_count: int,
        max_questions: int,
    ) -> str:
        _ = (ask_service, planning_request, max_questions)
        generated_args.append(
            {
                "prior_questions": list(prior_questions),
                "prior_answers": list(prior_answers),
                "asked_count": asked_count,
            }
        )
        return f"LLM question {asked_count + 1}?"

    monkeypatch.setattr("mana_analyzer.commands.cli._generate_planning_question_llm", _fake_llm_question)

    user_input = "\n".join(
        [
            "plan auth module",
            "answer one",
            "answer two",
            "answer three",
            "quit",
        ]
    ) + "\n"

    result = runner.invoke(
        cli.app,
        ["chat", "--planning-mode", "--planning-max-questions", "3"],
        input=user_input,
    )

    assert result.exit_code == 0
    assert "LLM question 1?" in result.stdout
    assert "LLM question 2?" in result.stdout
    assert "LLM question 3?" in result.stdout
    assert generated_args[1]["prior_answers"] == ["answer one"]
    assert generated_args[2]["prior_answers"] == ["answer one", "answer two"]


def test_chat_planning_mode_falls_back_to_static_on_llm_question_failure(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr(
        "mana_analyzer.commands.cli.build_ask_service",
        lambda _s, model_override=None: RecordingAskService(calls),
    )
    monkeypatch.setattr(
        "mana_analyzer.commands.cli._generate_planning_question_llm",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("llm question failed")),
    )

    user_input = "\n".join(
        [
            "plan auth module",
            "answer one",
            "answer two",
            "answer three",
            "quit",
        ]
    ) + "\n"

    result = runner.invoke(
        cli.app,
        ["chat", "--planning-mode", "--planning-max-questions", "3"],
        input=user_input,
    )

    assert result.exit_code == 0
    assert "What is the concrete goal and the success criteria?" in result.stdout
