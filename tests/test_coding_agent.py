from __future__ import annotations

from pathlib import Path

from mana_analyzer.llm.coding_agent import CodingAgent


class _Tool:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeAskAgent:
    def __init__(self, repo_root: Path, *, write_on_attempt: int | None) -> None:
        self.repo_root = repo_root
        self.write_on_attempt = write_on_attempt
        self.tools: list[object] = []
        self.questions: list[str] = []

    def run(
        self,
        question: str,
        index_dir: str | Path,
        k: int,
        max_steps: int,
        timeout_seconds: int,
        callbacks: list[object] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        _ = (index_dir, k, max_steps, timeout_seconds, callbacks, system_prompt)
        self.questions.append(question)
        if self.write_on_attempt is not None and len(self.questions) == self.write_on_attempt:
            target = self.repo_root / "src" / "fixed.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("def fixed() -> None:\n    pass\n", encoding="utf-8")
            return "Applied a fix."
        return "Reviewed context only."

    def ask(self, question: str, **kwargs: object) -> str:
        return self.run(question=question, index_dir=".", k=4, max_steps=6, timeout_seconds=30, callbacks=None)


def _wire_deterministic_status(agent: CodingAgent, repo_root: Path) -> None:
    agent._git_status_paths = lambda: {  # type: ignore[method-assign]
        p.relative_to(repo_root).as_posix() for p in repo_root.rglob("*") if p.is_file()
    }
    agent._git_diff = lambda _paths: ""  # type: ignore[method-assign]
    agent._run_static_analysis = lambda _py_paths: []  # type: ignore[method-assign]


def test_coding_agent_retries_edit_request_when_no_files_changed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "mana_analyzer.llm.coding_agent.build_write_file_tool",
        lambda **_kwargs: _Tool("write_file"),
    )
    monkeypatch.setattr(
        "mana_analyzer.llm.coding_agent.build_apply_patch_tool",
        lambda **_kwargs: _Tool("apply_patch"),
    )
    (tmp_path / "README.md").write_text("seed\n", encoding="utf-8")

    ask_agent = _FakeAskAgent(tmp_path, write_on_attempt=2)
    agent = CodingAgent(
        api_key="test-key",
        repo_root=tmp_path,
        ask_agent=ask_agent,
        allowed_prefixes=None,
    )
    _wire_deterministic_status(agent, tmp_path)

    result = agent.generate(
        "Fix the parser bug in src/mana_analyzer/services/ask_service.py",
        index_dir=tmp_path / ".mana_index",
        k=4,
    )

    assert len(ask_agent.questions) == 2
    assert "MANDATORY RETRY:" in ask_agent.questions[1]
    assert "src/fixed.py" in result["changed_files"]
    assert result["warnings"]
    assert "retrying with explicit edit instructions" in result["warnings"][0].lower()


def test_coding_agent_does_not_retry_read_only_prompt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "mana_analyzer.llm.coding_agent.build_write_file_tool",
        lambda **_kwargs: _Tool("write_file"),
    )
    monkeypatch.setattr(
        "mana_analyzer.llm.coding_agent.build_apply_patch_tool",
        lambda **_kwargs: _Tool("apply_patch"),
    )
    (tmp_path / "README.md").write_text("seed\n", encoding="utf-8")

    ask_agent = _FakeAskAgent(tmp_path, write_on_attempt=None)
    agent = CodingAgent(
        api_key="test-key",
        repo_root=tmp_path,
        ask_agent=ask_agent,
        allowed_prefixes=None,
    )
    _wire_deterministic_status(agent, tmp_path)

    result = agent.generate(
        "Explain how directory mode chooses indexes.",
        index_dir=tmp_path / ".mana_index",
        k=4,
    )

    assert len(ask_agent.questions) == 1
    assert result["changed_files"] == []
    assert result["warnings"] == []


def test_coding_agent_uses_write_file_fallback_retry(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "mana_analyzer.llm.coding_agent.build_write_file_tool",
        lambda **_kwargs: _Tool("write_file"),
    )
    monkeypatch.setattr(
        "mana_analyzer.llm.coding_agent.build_apply_patch_tool",
        lambda **_kwargs: _Tool("apply_patch"),
    )
    (tmp_path / "README.md").write_text("seed\n", encoding="utf-8")

    ask_agent = _FakeAskAgent(tmp_path, write_on_attempt=3)
    agent = CodingAgent(
        api_key="test-key",
        repo_root=tmp_path,
        ask_agent=ask_agent,
        allowed_prefixes=None,
    )
    _wire_deterministic_status(agent, tmp_path)

    result = agent.generate(
        "Fix the parser bug in src/mana_analyzer/services/ask_service.py",
        index_dir=tmp_path / ".mana_index",
        k=4,
    )

    assert len(ask_agent.questions) == 3
    assert "MANDATORY RETRY 2" in ask_agent.questions[2]
    assert "write_file" in ask_agent.questions[2]
    assert "Do NOT attempt another patch-only retry" in ask_agent.questions[2]
    assert "src/fixed.py" in result["changed_files"]
    assert any("switching to write_file fallback" in warning for warning in result["warnings"])


def test_coding_agent_stops_after_write_file_fallback_with_no_changes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "mana_analyzer.llm.coding_agent.build_write_file_tool",
        lambda **_kwargs: _Tool("write_file"),
    )
    monkeypatch.setattr(
        "mana_analyzer.llm.coding_agent.build_apply_patch_tool",
        lambda **_kwargs: _Tool("apply_patch"),
    )
    (tmp_path / "README.md").write_text("seed\n", encoding="utf-8")

    ask_agent = _FakeAskAgent(tmp_path, write_on_attempt=None)
    agent = CodingAgent(
        api_key="test-key",
        repo_root=tmp_path,
        ask_agent=ask_agent,
        allowed_prefixes=None,
    )
    _wire_deterministic_status(agent, tmp_path)

    result = agent.generate(
        "Fix the parser bug in src/mana_analyzer/services/ask_service.py",
        index_dir=tmp_path / ".mana_index",
        k=4,
    )

    assert len(ask_agent.questions) == 3
    assert result["changed_files"] == []
    assert any("switching to write_file fallback" in warning for warning in result["warnings"])
    assert any("Stopping retries to avoid patch-only loops" in warning for warning in result["warnings"])
