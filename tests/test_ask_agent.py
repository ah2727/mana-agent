from __future__ import annotations

import subprocess
from pathlib import Path

from langchain_core.tools import StructuredTool

from mana_analyzer.analysis.models import AskResponseWithTrace, SearchHit, ToolInvocationTrace
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


def test_ask_agent_run_multi_continues_when_presearch_has_no_hits(tmp_path: Path) -> None:
    class _NoHitSearchService:
        def search_multi(self, index_dirs: list[Path], query: str, k: int) -> tuple[list[SearchHit], list[str]]:
            _ = (index_dirs, query, k)
            return [], ["presearch warning"]

    agent = _build_agent(tmp_path)
    agent.search_service = _NoHitSearchService()
    first = tmp_path / "a" / ".mana_index"
    second = tmp_path / "b" / ".mana_index"
    captured: dict[str, object] = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return AskResponseWithTrace(
            answer="Created project scaffold.",
            sources=[],
            warnings=["agent warning"],
            mode="agent-tools",
            trace=[
                ToolInvocationTrace(
                    tool_name="run_command",
                    args_summary="cmd='mkdir src'",
                    duration_ms=1.0,
                    status="ok",
                    output_preview='{"returncode": 0}',
                )
            ],
        )

    agent.run = _fake_run  # type: ignore[method-assign]

    result = agent.run_multi("Create a NestJS project", [first, second], 3, max_steps=2, timeout_seconds=2)
    assert result.answer == "Created project scaffold."
    assert any("No indexed hits found across indexes; continuing with tool loop." in w for w in result.warnings)
    assert "presearch warning" in result.warnings
    assert "agent warning" in result.warnings
    assert captured["question"] == "Create a NestJS project"
    assert captured["index_dir"] == first.resolve()
    assert captured["index_dirs"] == [first.resolve(), second.resolve()]


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


def test_ask_agent_does_not_disable_search_internet_after_repeated_calls(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)
    agent.tools = [
        StructuredTool.from_function(
            func=lambda query: {"ok": True, "query": query, "results": [{"title": "x"}], "error": ""},
            name="search_internet",
            description="Search external information.",
        )
    ]
    agent.llm = _FakeLLM(
        [
            _FakeAIMessage("", tool_calls=[{"id": "1", "name": "search_internet", "args": {"query": "q1"}}]),
            _FakeAIMessage("", tool_calls=[{"id": "2", "name": "search_internet", "args": {"query": "q2"}}]),
            _FakeAIMessage("", tool_calls=[{"id": "3", "name": "search_internet", "args": {"query": "q3"}}]),
            _FakeAIMessage("Done", tool_calls=[]),
        ]
    )

    result = agent.run("Need latest info", tmp_path / ".mana_index", 2, max_steps=5, timeout_seconds=2)
    assert result.answer == "Done"
    assert not any("disabled after repeated calls without progress" in str(w) for w in result.warnings)


def test_ask_agent_reports_search_internet_failure_once(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)
    agent.tools = [
        StructuredTool.from_function(
            func=lambda query: {"ok": False, "query": query, "results": [], "error": "DuckDuckGo fallback failed"},
            name="search_internet",
            description="Search external information.",
        )
    ]
    agent.llm = _FakeLLM(
        [
            _FakeAIMessage("", tool_calls=[{"id": "1", "name": "search_internet", "args": {"query": "q1"}}]),
            _FakeAIMessage("", tool_calls=[{"id": "2", "name": "search_internet", "args": {"query": "q1"}}]),
            _FakeAIMessage("Done", tool_calls=[]),
        ]
    )

    result = agent.run("Need latest info", tmp_path / ".mana_index", 2, max_steps=4, timeout_seconds=2)
    assert result.answer == "Done"
    matches = [w for w in result.warnings if "search_internet failed: DuckDuckGo fallback failed" in str(w)]
    assert len(matches) == 1


def test_ask_agent_suppresses_missing_backend_search_warning(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)
    agent.tools = [
        StructuredTool.from_function(
            func=lambda query: {
                "ok": False,
                "query": query,
                "results": [],
                "error": "DuckDuckGo fallback failed (TAVILY_API_KEY not set)",
            },
            name="search_internet",
            description="Search external information.",
        )
    ]
    agent.llm = _FakeLLM(
        [
            _FakeAIMessage("", tool_calls=[{"id": "1", "name": "search_internet", "args": {"query": "q1"}}]),
            _FakeAIMessage("Done", tool_calls=[]),
        ]
    )

    result = agent.run("Need latest info", tmp_path / ".mana_index", 2, max_steps=3, timeout_seconds=2)
    assert result.answer == "Done"
    assert not any("search_internet failed:" in str(w) for w in result.warnings)


def test_is_apply_patch_failure_treats_ok_true_payload_with_error_details_as_success() -> None:
    payload = (
        '{"ok": true, "error": "", "attempts": ['
        '{"strategy":"git","phase":"check-p0","ok":false,"detail":"error: patch failed"}]}'
    )
    assert AskAgent._is_apply_patch_failure(payload) is False


def test_ask_agent_keeps_looping_after_apply_patch_failures_for_write_file_fallback(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)
    agent.tools = [
        StructuredTool.from_function(
            func=lambda diff: {"ok": False, "error": "hunk context mismatch", "touched_files": ["src/demo.py"]},
            name="apply_patch",
            description="Apply a patch.",
        ),
        StructuredTool.from_function(
            func=lambda path, content: {"ok": True, "path": path, "bytes": len(content)},
            name="write_file",
            description="Write a file.",
        ),
    ]
    agent.llm = _FakeLLM(
        [
            _FakeAIMessage("", tool_calls=[{"id": "1", "name": "apply_patch", "args": {"diff": "d1"}}]),
            _FakeAIMessage("", tool_calls=[{"id": "2", "name": "apply_patch", "args": {"diff": "d2"}}]),
            _FakeAIMessage(
                "",
                tool_calls=[{"id": "3", "name": "write_file", "args": {"path": "src/demo.py", "content": "print(1)\n"}}],
            ),
            _FakeAIMessage("Done", tool_calls=[]),
        ]
    )

    result = agent.run("Implement change", tmp_path / ".mana_index", 2, max_steps=6, timeout_seconds=2)
    assert result.answer == "Done"
    assert any("apply_patch disabled after repeated failures" in str(w) for w in result.warnings)
