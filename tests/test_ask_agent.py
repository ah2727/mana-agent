from __future__ import annotations

import subprocess
from pathlib import Path

from langchain_core.tools import StructuredTool

from mana_analyzer.analysis.models import SearchHit
from mana_analyzer.llm.ask_agent import AskAgent


class _FakeSearchService:
    def search(self, index_dir: Path, query: str, k: int) -> list[SearchHit]:
        assert index_dir
        assert query
        assert k > 0
        return [
            SearchHit(
                score=0.91,
                file_path="/tmp/example.py",
                start_line=3,
                end_line=8,
                symbol_name="demo",
                snippet="def demo(): pass",
            )
        ]


class _FakeAIMessage:
    def __init__(self, content: str, tool_calls: list[dict] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeBoundModel:
    def __init__(self, responses: list[_FakeAIMessage]) -> None:
        self._responses = responses
        self._idx = 0

    def invoke(self, _messages: list[object]) -> _FakeAIMessage:
        value = self._responses[self._idx]
        self._idx += 1
        return value


class _FakeLLM:
    def __init__(self, responses: list[_FakeAIMessage]) -> None:
        self._responses = responses

    def bind_tools(self, _tools: list[object]) -> _FakeBoundModel:
        return _FakeBoundModel(self._responses)


def _build_agent(tmp_path: Path) -> AskAgent:
    agent = AskAgent.__new__(AskAgent)
    agent.search_service = _FakeSearchService()
    agent.project_root = tmp_path.resolve()
    agent._resolved_index = tmp_path / ".mana_index"
    return agent


def test_ask_agent_enforces_max_steps(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)
    agent.llm = _FakeLLM(
        [
            _FakeAIMessage(
                "",
                tool_calls=[{"id": "1", "name": "semantic_search", "args": {"query": "find x", "k": 2}}],
            ),
            _FakeAIMessage(
                "",
                tool_calls=[{"id": "2", "name": "semantic_search", "args": {"query": "find y", "k": 2}}],
            ),
        ]
    )
    result = agent.run("How?", tmp_path / ".mana_index", 2, max_steps=1, timeout_seconds=2)
    assert "step limit" in result.answer.lower()
    assert result.trace


def test_ask_agent_blocks_dangerous_command(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)
    tools, traces, _, _ = agent._build_tools(k_default=4, timeout_seconds=1)
    run_command = [item for item in tools if item.name == "run_command"][0]

    output = run_command.invoke({"cmd": "rm -rf /tmp/foo"})
    assert "blocked" in output.lower()
    assert traces[-1].status == "error"


def test_ask_agent_records_timeout(tmp_path: Path, monkeypatch) -> None:
    agent = _build_agent(tmp_path)
    tools, traces, _, _ = agent._build_tools(k_default=4, timeout_seconds=1)
    run_command = [item for item in tools if item.name == "run_command"][0]

    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="rg x", timeout=1)

    monkeypatch.setattr("mana_analyzer.llm.ask_agent.subprocess.run", _raise_timeout)
    output = run_command.invoke({"cmd": "rg demo"})
    assert "timed out" in output.lower()
    assert traces[-1].status == "timeout"


def test_ask_agent_collects_trace_and_sources(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)
    agent.llm = _FakeLLM(
        [
            _FakeAIMessage(
                "",
                tool_calls=[{"id": "1", "name": "semantic_search", "args": {"query": "demo", "k": 3}}],
            ),
            _FakeAIMessage("Answer with /tmp/example.py:3-8", tool_calls=[]),
        ]
    )
    result = agent.run("Where is demo?", tmp_path / ".mana_index", 3, max_steps=3, timeout_seconds=2)
    assert "example.py:3-8" in result.answer
    assert result.sources
    assert any(item.tool_name == "semantic_search" for item in result.trace)


def test_ask_agent_run_multi_uses_all_indexes(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)
    agent.llm = _FakeLLM([_FakeAIMessage("Answer with citations", tool_calls=[])])
    first = tmp_path / "a" / ".mana_index"
    second = tmp_path / "b" / ".mana_index"
    result = agent.run_multi("Where?", [first, second], 3, max_steps=2, timeout_seconds=2)
    assert result.mode == "agent-tools"
    assert len(agent._resolved_indexes) == 2


def test_ask_agent_invokes_externally_registered_tool(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)
    agent.tools = [
        StructuredTool.from_function(
            func=lambda query: f'{{"ok": true, "query": "{query}"}}',
            name="search_internet",
            description="Search external information.",
        )
    ]
    agent.llm = _FakeLLM(
        [
            _FakeAIMessage(
                "",
                tool_calls=[{"id": "1", "name": "search_internet", "args": {"query": "latest"}}],
            ),
            _FakeAIMessage("Done", tool_calls=[]),
        ]
    )

    result = agent.run("Need latest info", tmp_path / ".mana_index", 2, max_steps=3, timeout_seconds=2)
    assert result.answer == "Done"
