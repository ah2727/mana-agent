from __future__ import annotations

import json
import subprocess
from pathlib import Path

from langchain_core.tools import StructuredTool

from mana_analyzer.analysis.models import AskResponseWithTrace, SearchHit, ToolInvocationTrace
from mana_analyzer.llm.ask_agent import AskAgent
from mana_analyzer.services.coding_memory_service import CodingMemoryService


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


def test_ask_agent_run_command_rewrites_python_to_local_venv_python3(tmp_path: Path, monkeypatch) -> None:
    agent = _build_agent(tmp_path)
    venv_python = tmp_path / ".venv" / "bin" / "python3"
    venv_python.parent.mkdir(parents=True, exist_ok=True)
    venv_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    venv_python.chmod(0o755)

    tools, _traces, _, _ = agent._build_tools(k_default=4, timeout_seconds=1)
    run_command = [item for item in tools if item.name == "run_command"][0]

    captured: dict[str, str] = {}

    def _fake_run(cmd, **kwargs):  # noqa: ANN001
        captured["cmd"] = str(cmd)
        _ = kwargs
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    monkeypatch.setattr("mana_analyzer.llm.ask_agent.subprocess.run", _fake_run)
    payload = json.loads(run_command.invoke({"cmd": "python -V"}))

    assert payload["interpreter_rewritten"] is True
    assert payload["original_cmd"] == "python -V"
    assert payload["executed_cmd"].startswith(str(venv_python))
    assert captured["cmd"].startswith(str(venv_python))


def test_ask_agent_run_command_rewrites_python_to_python3_without_local_venv(tmp_path: Path, monkeypatch) -> None:
    agent = _build_agent(tmp_path)
    tools, _traces, _, _ = agent._build_tools(k_default=4, timeout_seconds=1)
    run_command = [item for item in tools if item.name == "run_command"][0]

    captured: dict[str, str] = {}

    def _fake_run(cmd, **kwargs):  # noqa: ANN001
        captured["cmd"] = str(cmd)
        _ = kwargs
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    monkeypatch.setattr("mana_analyzer.llm.ask_agent.subprocess.run", _fake_run)
    payload = json.loads(run_command.invoke({"cmd": "python -m pytest -q"}))

    assert payload["interpreter_rewritten"] is True
    assert payload["executed_cmd"].startswith("python3 ")
    assert payload["original_cmd"] == "python -m pytest -q"
    assert captured["cmd"].startswith("python3 ")


def test_ask_agent_read_file_uses_policy_line_window(tmp_path: Path) -> None:
    source_file = tmp_path / "src.py"
    source_file.write_text("\n".join(f"line-{idx}" for idx in range(1, 1500)), encoding="utf-8")
    agent = _build_agent(tmp_path)
    tools, _traces, _, _ = agent._build_tools(k_default=4, timeout_seconds=1, read_line_window=900)
    read_file = [item for item in tools if item.name == "read_file"][0]

    payload = json.loads(read_file.invoke({"path": str(source_file), "start_line": 10, "end_line": 5000}))
    assert int(payload["start_line"]) == 10
    assert int(payload["end_line"]) == 910


def test_ask_agent_read_file_line_window_is_safely_clamped(tmp_path: Path) -> None:
    source_file = tmp_path / "src.py"
    source_file.write_text("\n".join(f"line-{idx}" for idx in range(1, 1500)), encoding="utf-8")
    agent = _build_agent(tmp_path)
    tools, _traces, _, _ = agent._build_tools(k_default=4, timeout_seconds=1, read_line_window=20)
    read_file = [item for item in tools if item.name == "read_file"][0]

    payload = json.loads(read_file.invoke({"path": str(source_file), "start_line": 1, "end_line": 5000}))
    assert int(payload["start_line"]) == 1
    assert int(payload["end_line"]) == 201


def test_ask_agent_read_file_full_mode_returns_entire_small_file(tmp_path: Path) -> None:
    source_file = tmp_path / "small.py"
    source_file.write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
    agent = _build_agent(tmp_path)
    tools, _traces, _, _ = agent._build_tools(k_default=4, timeout_seconds=1)
    read_file = [item for item in tools if item.name == "read_file"][0]

    payload = json.loads(read_file.invoke({"path": str(source_file), "mode": "full"}))
    assert payload["mode"] == "full"
    assert payload["cache_hit"] is False
    assert payload["cache_source"] == "disk"
    assert payload["full_file_cached"] is True
    assert payload["line_count"] == 3
    assert payload["content"] == "a = 1\nb = 2\nc = 3\n"


def test_ask_agent_read_file_full_mode_oversized_returns_structured_error(tmp_path: Path) -> None:
    source_file = tmp_path / "big.py"
    source_file.write_text("\n".join(f"line-{idx}" for idx in range(6001)), encoding="utf-8")
    agent = _build_agent(tmp_path)
    tools, _traces, _, _ = agent._build_tools(k_default=4, timeout_seconds=1)
    read_file = [item for item in tools if item.name == "read_file"][0]

    payload = json.loads(read_file.invoke({"path": str(source_file), "mode": "full"}))
    assert "use mode='line'" in payload["error"]
    assert payload["mode"] == "full"
    assert payload["line_count"] == 6001
    assert payload["max_lines"] == AskAgent.READ_FULL_FILE_MAX_LINES


def test_ask_agent_read_file_full_mode_hits_persistent_flow_cache_on_repeat(tmp_path: Path) -> None:
    source_file = tmp_path / "cached.py"
    source_file.write_text("one\ntwo\nthree\n", encoding="utf-8")
    service = CodingMemoryService(project_root=tmp_path)

    first = _build_agent(tmp_path)
    first.coding_memory_service = service
    tools1, _traces1, _, _ = first._build_tools(k_default=4, timeout_seconds=1, flow_id="flow-cache-1")
    read_file1 = [item for item in tools1 if item.name == "read_file"][0]
    first_payload = json.loads(read_file1.invoke({"path": str(source_file), "mode": "full"}))

    second = _build_agent(tmp_path)
    second.coding_memory_service = service
    tools2, _traces2, _, _ = second._build_tools(k_default=4, timeout_seconds=1, flow_id="flow-cache-1")
    read_file2 = [item for item in tools2 if item.name == "read_file"][0]
    second_payload = json.loads(read_file2.invoke({"path": str(source_file), "mode": "full"}))

    assert first_payload["cache_hit"] is False
    assert second_payload["cache_hit"] is True
    assert second_payload["cache_source"] == "flow_full"


def test_ask_agent_read_file_line_mode_uses_full_cache_slice(tmp_path: Path) -> None:
    source_file = tmp_path / "slice.py"
    source_file.write_text("\n".join(f"line-{idx}" for idx in range(1, 8)) + "\n", encoding="utf-8")
    service = CodingMemoryService(project_root=tmp_path)
    agent = _build_agent(tmp_path)
    agent.coding_memory_service = service
    telemetry = {
        "read_cache_hits": 0,
        "read_cache_misses": 0,
        "read_full_mode_used": 0,
        "read_full_mode_blocked": 0,
        "read_cache_invalidations": 0,
    }
    ephemeral: dict[str, list[dict[str, object]]] = {}
    tools, _traces, _, _ = agent._build_tools(
        k_default=4,
        timeout_seconds=1,
        flow_id="flow-cache-2",
        ephemeral_read_cache=ephemeral,
        read_telemetry=telemetry,
    )
    read_file = [item for item in tools if item.name == "read_file"][0]

    full_payload = json.loads(read_file.invoke({"path": str(source_file), "mode": "full"}))
    line_payload = json.loads(
        read_file.invoke({"path": str(source_file), "mode": "line", "start_line": 2, "end_line": 4})
    )

    assert full_payload["cache_hit"] is False
    assert line_payload["cache_hit"] is True
    assert line_payload["cache_source"] == "flow_full"
    assert line_payload["content"] == "line-2\nline-3\nline-4"
    assert telemetry["read_cache_hits"] == 1


def test_ask_agent_read_file_cache_invalidates_when_file_changes(tmp_path: Path) -> None:
    source_file = tmp_path / "stale.py"
    source_file.write_text("old-1\nold-2\n", encoding="utf-8")
    service = CodingMemoryService(project_root=tmp_path)
    agent = _build_agent(tmp_path)
    agent.coding_memory_service = service
    telemetry = {
        "read_cache_hits": 0,
        "read_cache_misses": 0,
        "read_full_mode_used": 0,
        "read_full_mode_blocked": 0,
        "read_cache_invalidations": 0,
    }
    tools, _traces, _, _ = agent._build_tools(
        k_default=4,
        timeout_seconds=1,
        flow_id="flow-cache-3",
        read_telemetry=telemetry,
    )
    read_file = [item for item in tools if item.name == "read_file"][0]

    first_payload = json.loads(read_file.invoke({"path": str(source_file), "mode": "full"}))
    source_file.write_text("new-1\nnew-2\nnew-3\n", encoding="utf-8")
    second_payload = json.loads(read_file.invoke({"path": str(source_file), "mode": "full"}))

    assert first_payload["cache_hit"] is False
    assert second_payload["cache_hit"] is False
    assert second_payload["cache_invalidated"] is True
    assert telemetry["read_cache_invalidations"] == 1
    assert second_payload["content"] == "new-1\nnew-2\nnew-3\n"


def test_ask_agent_read_file_without_flow_id_uses_only_ephemeral_cache(tmp_path: Path) -> None:
    source_file = tmp_path / "ephemeral.py"
    source_file.write_text("x\ny\n", encoding="utf-8")
    service = CodingMemoryService(project_root=tmp_path)

    first = _build_agent(tmp_path)
    first.coding_memory_service = service
    tools1, _traces1, _, _ = first._build_tools(k_default=4, timeout_seconds=1)
    read_file1 = [item for item in tools1 if item.name == "read_file"][0]
    payload1 = json.loads(read_file1.invoke({"path": str(source_file), "mode": "full"}))

    second = _build_agent(tmp_path)
    second.coding_memory_service = service
    tools2, _traces2, _, _ = second._build_tools(k_default=4, timeout_seconds=1)
    read_file2 = [item for item in tools2 if item.name == "read_file"][0]
    payload2 = json.loads(read_file2.invoke({"path": str(source_file), "mode": "full"}))

    assert payload1["cache_hit"] is False
    assert payload2["cache_hit"] is False


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


def test_ask_agent_extracts_text_from_list_content_blocks(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)
    agent.llm = _FakeLLM(
        [
            _FakeAIMessage(
                [
                    {"id": "rs_1", "summary": [], "type": "reasoning"},
                    {"type": "text", "text": "Only the final answer should be shown."},
                ],
                tool_calls=[],
            ),
        ]
    )

    result = agent.run("Why?", tmp_path / ".mana_index", 3, max_steps=2, timeout_seconds=2)
    assert result.answer == "Only the final answer should be shown."


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
