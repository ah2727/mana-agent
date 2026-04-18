from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from mana_analyzer.llm import tool_worker_process as twp


class _FakeStdout:
    def __init__(self) -> None:
        self.lines: deque[str] = deque()

    def readline(self) -> str:
        if not self.lines:
            return ""
        return self.lines.popleft()

    def push(self, payload: dict) -> None:
        self.lines.append(json.dumps(payload) + "\n")


class _FakeStdin:
    def __init__(self, on_write) -> None:
        self._on_write = on_write

    def write(self, text: str) -> int:
        self._on_write(text)
        return len(text)

    def flush(self) -> None:
        return None


class _FakeProc:
    def __init__(self, handler) -> None:
        self.stdout = _FakeStdout()
        self.stdin = _FakeStdin(handler)
        self._alive = True
        self._handler = handler

    def poll(self):
        return None if self._alive else 1

    def terminate(self) -> None:
        self._alive = False

    def wait(self, timeout: float | None = None) -> int:
        _ = timeout
        self._alive = False
        return 0

    def kill(self) -> None:
        self._alive = False


def _reply_ok(request_id: str, payload: dict) -> dict:
    return {"type": "ok", "request_id": request_id, "payload": payload}


def _reply_error(request_id: str, code: str, message: str, retriable: bool = False) -> dict:
    return {
        "type": "error",
        "request_id": request_id,
        "payload": {"code": code, "message": message, "retriable": retriable, "details": {}},
    }


def test_tool_worker_client_init_health_shutdown(monkeypatch) -> None:
    proc_holder = {}

    def _make_proc(*_args, **_kwargs):
        def _handle_write(text: str) -> None:
            line = text.strip()
            if not line:
                return
            req = json.loads(line)
            rid = req["request_id"]
            kind = req["type"]
            if kind == "init":
                proc.stdout.push({"type": "event", "request_id": rid, "payload": {"name": "initialized"}})
                proc.stdout.push(_reply_ok(rid, {"status": "ok"}))
            elif kind == "health":
                proc.stdout.push(_reply_ok(rid, {"status": "ok"}))
            elif kind == "shutdown":
                proc.stdout.push(_reply_ok(rid, {"status": "bye"}))
                proc._alive = False

        proc = _FakeProc(_handle_write)
        proc_holder["proc"] = proc
        return proc

    monkeypatch.setattr(twp.subprocess, "Popen", _make_proc)
    client = twp.ToolWorkerClient(
        api_key="x",
        model="fake-model",
        repo_root=Path("/tmp"),
        project_root=Path("/tmp"),
    )
    client.start()
    health = client.health()
    assert health["status"] == "ok"
    client.stop()
    assert proc_holder["proc"].poll() is not None


def test_tool_worker_client_restarts_once_on_worker_failure(monkeypatch) -> None:
    procs: list[_FakeProc] = []

    def _make_proc(*_args, **_kwargs):
        idx = len(procs)

        def _handle_write(text: str) -> None:
            req = json.loads(text.strip())
            rid = req["request_id"]
            kind = req["type"]
            if kind == "init":
                proc.stdout.push({"type": "event", "request_id": rid, "payload": {"name": "initialized"}})
                proc.stdout.push(_reply_ok(rid, {"status": "ok"}))
                return
            if kind == "run_tools":
                if idx == 0:
                    proc._alive = False
                    return
                proc.stdout.push(
                    _reply_ok(
                        rid,
                        {
                            "answer": "ok",
                            "sources": [],
                            "mode": "agent-tools",
                            "trace": [{"tool_name": "read_file", "status": "ok"}],
                            "warnings": [],
                        },
                    )
                )
                return
            if kind == "shutdown":
                proc.stdout.push(_reply_ok(rid, {"status": "bye"}))
                proc._alive = False

        proc = _FakeProc(_handle_write)
        procs.append(proc)
        return proc

    monkeypatch.setattr(twp.subprocess, "Popen", _make_proc)
    client = twp.ToolWorkerClient(
        api_key="x",
        model="fake-model",
        repo_root=Path("/tmp"),
        project_root=Path("/tmp"),
    )
    response = client.run_tools(
        twp.ToolRunRequest(
            question="q",
            index_dir="/tmp/.mana/index",
            k=4,
            max_steps=4,
            timeout_seconds=5,
        )
    )
    assert response.answer == "ok"
    assert len(procs) == 2
    client.stop()


def test_tool_worker_client_run_tools_forwards_events(monkeypatch) -> None:
    def _make_proc(*_args, **_kwargs):
        def _handle_write(text: str) -> None:
            req = json.loads(text.strip())
            rid = req["request_id"]
            kind = req["type"]
            if kind == "init":
                proc.stdout.push({"type": "event", "request_id": rid, "payload": {"name": "initialized"}})
                proc.stdout.push(_reply_ok(rid, {"status": "ok"}))
                return
            if kind == "run_tools":
                proc.stdout.push(
                    {
                        "type": "event",
                        "request_id": rid,
                        "payload": {
                            "name": "tool_start",
                            "message": "TOOL start: read_file | args: {'path':'x.py'}",
                            "data": {"tool": "read_file", "args": "{'path':'x.py'}"},
                        },
                    }
                )
                proc.stdout.push(
                    _reply_ok(
                        rid,
                        {
                            "answer": "ok",
                            "sources": [],
                            "mode": "agent-tools",
                            "trace": [{"tool_name": "read_file", "status": "ok"}],
                            "warnings": [],
                        },
                    )
                )
                return
            if kind == "shutdown":
                proc.stdout.push(_reply_ok(rid, {"status": "bye"}))
                proc._alive = False

        proc = _FakeProc(_handle_write)
        return proc

    monkeypatch.setattr(twp.subprocess, "Popen", _make_proc)
    client = twp.ToolWorkerClient(
        api_key="x",
        model="fake-model",
        repo_root=Path("/tmp"),
        project_root=Path("/tmp"),
    )

    seen_names: list[str] = []
    response = client.run_tools(
        twp.ToolRunRequest(
            question="q",
            index_dir="/tmp/.mana/index",
            k=4,
            max_steps=4,
            timeout_seconds=5,
        ),
        on_event=lambda event: seen_names.append(event.name),
    )

    assert response.answer == "ok"
    assert "tool_start" in seen_names
    client.stop()


def test_tool_worker_client_does_not_retry_non_retriable_run_failed(monkeypatch) -> None:
    run_tools_calls = 0

    def _make_proc(*_args, **_kwargs):
        def _handle_write(text: str) -> None:
            nonlocal run_tools_calls
            req = json.loads(text.strip())
            rid = req["request_id"]
            kind = req["type"]
            if kind == "init":
                proc.stdout.push({"type": "event", "request_id": rid, "payload": {"name": "initialized"}})
                proc.stdout.push(_reply_ok(rid, {"status": "ok"}))
                return
            if kind == "run_tools":
                run_tools_calls += 1
                proc.stdout.push(
                    _reply_error(
                        rid,
                        "run_failed",
                        "Error code: 400 - {'error': {'message': 'openai_error', 'type': 'bad_response_status_code'}}",
                        retriable=False,
                    )
                )
                return
            if kind == "shutdown":
                proc.stdout.push(_reply_ok(rid, {"status": "bye"}))
                proc._alive = False

        proc = _FakeProc(_handle_write)
        return proc

    monkeypatch.setattr(twp.subprocess, "Popen", _make_proc)
    client = twp.ToolWorkerClient(
        api_key="x",
        model="fake-model",
        repo_root=Path("/tmp"),
        project_root=Path("/tmp"),
    )

    with pytest.raises(twp.ToolWorkerProcessError) as excinfo:
        client.run_tools(
            twp.ToolRunRequest(
                question="q",
                index_dir="/tmp/.mana/index",
                k=4,
                max_steps=4,
                timeout_seconds=5,
            )
        )
    assert excinfo.value.code == "run_failed"
    assert run_tools_calls == 1
    client.stop()


def test_tool_worker_server_enforces_tools_only_violation(monkeypatch) -> None:
    class _FakeAskAgent:
        def run(self, **_kwargs):
            return SimpleNamespace(answer="no tools", sources=[], mode="agent-tools", trace=[], warnings=[])

    server = twp._ToolWorkerServer()
    server._ask_agent = _FakeAskAgent()  # type: ignore[assignment]
    server._tools_only_strict = True
    emitted: list[twp.WorkerReply] = []
    monkeypatch.setattr(twp._ToolWorkerServer, "_emit", staticmethod(lambda reply: emitted.append(reply)))
    server._handle_run_tools(
        twp.WorkerEnvelope(
            type="run_tools",
            request_id="req-1",
            payload=twp.ToolRunRequest(question="x", index_dir="/tmp/.mana/index").model_dump(),
        )
    )
    assert emitted
    assert emitted[-1].type == "error"
    assert emitted[-1].payload["code"] == "tools_only_violation"


def test_tool_worker_server_accepts_successful_tool_trace(monkeypatch) -> None:
    class _TraceRow:
        def to_dict(self) -> dict:
            return {"tool_name": "read_file", "status": "ok", "duration_ms": 1.0}

    class _FakeAskAgent:
        def run(self, **_kwargs):
            return SimpleNamespace(
                answer="done",
                sources=[],
                mode="agent-tools",
                trace=[_TraceRow()],
                warnings=[],
            )

    server = twp._ToolWorkerServer()
    server._ask_agent = _FakeAskAgent()  # type: ignore[assignment]
    server._tools_only_strict = True
    emitted: list[twp.WorkerReply] = []
    monkeypatch.setattr(twp._ToolWorkerServer, "_emit", staticmethod(lambda reply: emitted.append(reply)))
    server._handle_run_tools(
        twp.WorkerEnvelope(
            type="run_tools",
            request_id="req-2",
            payload=twp.ToolRunRequest(question="x", index_dir="/tmp/.mana/index").model_dump(),
        )
    )
    assert emitted
    assert emitted[-1].type == "ok"
    assert emitted[-1].payload["answer"] == "done"


def test_tool_worker_server_allows_no_tool_success_when_override_disabled(monkeypatch) -> None:
    class _FakeAskAgent:
        def run(self, **_kwargs):
            return SimpleNamespace(answer="no tools", sources=[], mode="agent-tools", trace=[], warnings=[])

    server = twp._ToolWorkerServer()
    server._ask_agent = _FakeAskAgent()  # type: ignore[assignment]
    server._tools_only_strict = True
    emitted: list[twp.WorkerReply] = []
    monkeypatch.setattr(twp._ToolWorkerServer, "_emit", staticmethod(lambda reply: emitted.append(reply)))
    server._handle_run_tools(
        twp.WorkerEnvelope(
            type="run_tools",
            request_id="req-override",
            payload=twp.ToolRunRequest(
                question="x",
                index_dir="/tmp/.mana/index",
                tools_only_strict_override=False,
            ).model_dump(),
        )
    )
    assert emitted
    assert emitted[-1].type == "ok"
    assert emitted[-1].payload["answer"] == "no tools"


def test_tool_worker_server_emits_tool_events(monkeypatch) -> None:
    class _TraceRow:
        def to_dict(self) -> dict:
            return {"tool_name": "read_file", "status": "ok", "duration_ms": 1.0}

    class _FakeAskAgent:
        def run(self, **kwargs):
            callbacks = kwargs.get("callbacks") or []
            if callbacks:
                cb = callbacks[0]
                cb.on_tool_start({"name": "read_file"}, '{"path":"x.py"}')
                cb.on_tool_end('{"ok":true}')
            return SimpleNamespace(
                answer="done",
                sources=[],
                mode="agent-tools",
                trace=[_TraceRow()],
                warnings=[],
            )

    server = twp._ToolWorkerServer()
    server._ask_agent = _FakeAskAgent()  # type: ignore[assignment]
    server._tools_only_strict = True
    emitted: list[twp.WorkerReply] = []
    monkeypatch.setattr(twp._ToolWorkerServer, "_emit", staticmethod(lambda reply: emitted.append(reply)))
    server._handle_run_tools(
        twp.WorkerEnvelope(
            type="run_tools",
            request_id="req-events",
            payload=twp.ToolRunRequest(question="x", index_dir="/tmp/.mana/index").model_dump(),
        )
    )

    event_names = [str(reply.payload.get("name", "")) for reply in emitted if reply.type == "event"]
    assert "tool_start" in event_names
    assert "tool_end" in event_names
    assert emitted[-1].type == "ok"


def test_tool_worker_server_blocks_duplicate_tool_name_within_turn(monkeypatch) -> None:
    calls: list[dict] = []

    class _TraceRow:
        def to_dict(self) -> dict:
            return {"tool_name": "read_file", "status": "ok", "duration_ms": 1.0}

    class _FakeAskAgent:
        def run(self, **_kwargs):
            calls.append({"called": True})
            return SimpleNamespace(
                answer="done",
                sources=[],
                mode="agent-tools",
                trace=[_TraceRow()],
                warnings=[],
            )

    server = twp._ToolWorkerServer()
    server._ask_agent = _FakeAskAgent()  # type: ignore[assignment]
    emitted: list[twp.WorkerReply] = []
    monkeypatch.setattr(twp._ToolWorkerServer, "_emit", staticmethod(lambda reply: emitted.append(reply)))

    first_payload = twp.ToolRunRequest(
        question="run once",
        index_dir="/tmp/.mana/index",
        tool_name="read_file",
    ).model_dump()
    second_payload = twp.ToolRunRequest(
        question="run twice",
        index_dir="/tmp/.mana/index",
        tool_name="read_file",
    ).model_dump()

    server._handle_run_tools(twp.WorkerEnvelope(type="run_tools", request_id="turn-1", payload=first_payload))
    server._handle_run_tools(twp.WorkerEnvelope(type="run_tools", request_id="turn-1", payload=second_payload))

    assert len(calls) == 1
    assert emitted[-1].type == "ok"
    assert emitted[-1].payload["answer"] == "Tool already executed in this turn."
    assert emitted[-1].payload["trace"][0]["status"] == "duplicate_blocked"


def test_tool_worker_server_allows_same_tool_name_in_new_turn(monkeypatch) -> None:
    calls: list[dict] = []

    class _TraceRow:
        def to_dict(self) -> dict:
            return {"tool_name": "read_file", "status": "ok", "duration_ms": 1.0}

    class _FakeAskAgent:
        def run(self, **_kwargs):
            calls.append({"called": True})
            return SimpleNamespace(
                answer="done",
                sources=[],
                mode="agent-tools",
                trace=[_TraceRow()],
                warnings=[],
            )

    server = twp._ToolWorkerServer()
    server._ask_agent = _FakeAskAgent()  # type: ignore[assignment]
    emitted: list[twp.WorkerReply] = []
    monkeypatch.setattr(twp._ToolWorkerServer, "_emit", staticmethod(lambda reply: emitted.append(reply)))

    payload = twp.ToolRunRequest(
        question="run",
        index_dir="/tmp/.mana/index",
        tool_name="read_file",
    ).model_dump()

    server._handle_run_tools(twp.WorkerEnvelope(type="run_tools", request_id="turn-a", payload=payload))
    server._turn_tool_state.reset()
    server._handle_run_tools(twp.WorkerEnvelope(type="run_tools", request_id="turn-b", payload=payload))

    assert len(calls) == 2


def test_tool_worker_server_marks_bad_request_as_non_retriable(monkeypatch) -> None:
    class _FakeAskAgent:
        def run(self, **_kwargs):
            raise RuntimeError(
                "Error code: 400 - {'error': {'message': 'openai_error', 'type': 'bad_response_status_code'}}"
            )

    server = twp._ToolWorkerServer()
    server._ask_agent = _FakeAskAgent()  # type: ignore[assignment]
    emitted: list[twp.WorkerReply] = []
    monkeypatch.setattr(twp._ToolWorkerServer, "_emit", staticmethod(lambda reply: emitted.append(reply)))

    server._handle_run_tools(
        twp.WorkerEnvelope(
            type="run_tools",
            request_id="req-bad-400",
            payload=twp.ToolRunRequest(question="x", index_dir="/tmp/.mana/index").model_dump(),
        )
    )

    assert emitted
    assert emitted[-1].type == "error"
    assert emitted[-1].payload["code"] == "run_failed"
    assert emitted[-1].payload["retriable"] is False


def test_tool_worker_server_marks_rate_limit_as_retriable(monkeypatch) -> None:
    class _FakeAskAgent:
        def run(self, **_kwargs):
            raise RuntimeError("Error code: 429 - rate limit exceeded")

    server = twp._ToolWorkerServer()
    server._ask_agent = _FakeAskAgent()  # type: ignore[assignment]
    emitted: list[twp.WorkerReply] = []
    monkeypatch.setattr(twp._ToolWorkerServer, "_emit", staticmethod(lambda reply: emitted.append(reply)))

    server._handle_run_tools(
        twp.WorkerEnvelope(
            type="run_tools",
            request_id="req-rate-429",
            payload=twp.ToolRunRequest(question="x", index_dir="/tmp/.mana/index").model_dump(),
        )
    )

    assert emitted
    assert emitted[-1].type == "error"
    assert emitted[-1].payload["code"] == "run_failed"
    assert emitted[-1].payload["retriable"] is True


def test_sanitize_openai_json_payload_normalizes_chat_messages() -> None:
    payload = {
        "model": "x",
        "messages": [
            {
                "role": "ai",
                "content": {"nested": "value"},
                "kwargs": {"junk": True},
                "status": "completed",
            },
            {
                "role": "assistant_tool",
                "content": {"ok": True},
                "tool_call_id": 123,
                "trace": [{"a": 1}],
            },
        ],
    }

    out = twp._sanitize_openai_json_payload(payload)
    assert isinstance(out, dict)
    messages = out["messages"]
    assert messages[0]["role"] == "assistant"
    assert isinstance(messages[0]["content"], str)
    assert "kwargs" not in messages[0]
    assert "status" not in messages[0]
    assert messages[1]["role"] == "tool"
    assert messages[1]["tool_call_id"] == "123"
    assert isinstance(messages[1]["content"], str)


def test_sanitize_openai_json_payload_normalizes_tool_calls_and_tools() -> None:
    payload = {
        "model": "x",
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": 99,
                        "type": "function",
                        "function": {"name": "read_file", "arguments": {"path": "README.md"}},
                        "extra": "drop",
                    }
                ],
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": {"bad": "shape"},
                    "parameters": {"type": "object"},
                    "strict": True,
                },
                "metadata": {"bad": True},
            }
        ],
    }

    out = twp._sanitize_openai_json_payload(payload)
    msg = out["messages"][0]
    assert msg["role"] == "assistant"
    assert msg["content"] == ""
    assert msg["tool_calls"][0]["id"] == "99"
    assert msg["tool_calls"][0]["function"]["arguments"] == '{"path": "README.md"}'
    tool = out["tools"][0]
    assert tool == {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "{'bad': 'shape'}",
            "parameters": {"type": "object"},
        },
    }


def test_run_tool_request_once_enforces_tools_only_policy(monkeypatch) -> None:
    class _FakeAskAgent:
        def run(self, **_kwargs):
            return SimpleNamespace(answer="no tools", sources=[], mode="agent-tools", trace=[], warnings=[])

    monkeypatch.setattr(twp, "_build_worker_ask_agent", lambda _payload: _FakeAskAgent())
    init_payload = twp.WorkerInitPayload(
        api_key="x",
        model="m",
        project_root="/tmp",
        repo_root="/tmp",
        tools_only_strict=True,
    )
    req = twp.ToolRunRequest(question="q", index_dir="/tmp/.mana/index")

    with pytest.raises(twp.ToolWorkerProcessError) as excinfo:
        twp.run_tool_request_once(init_payload=init_payload, request=req)
    assert excinfo.value.code == "tools_only_violation"


def test_run_tool_request_once_respects_tools_only_override(monkeypatch) -> None:
    class _FakeAskAgent:
        def run(self, **_kwargs):
            return SimpleNamespace(answer="ok", sources=[], mode="agent-tools", trace=[], warnings=[])

    monkeypatch.setattr(twp, "_build_worker_ask_agent", lambda _payload: _FakeAskAgent())
    init_payload = twp.WorkerInitPayload(
        api_key="x",
        model="m",
        project_root="/tmp",
        repo_root="/tmp",
        tools_only_strict=True,
    )
    req = twp.ToolRunRequest(
        question="q",
        index_dir="/tmp/.mana/index",
        tools_only_strict_override=False,
    )

    response = twp.run_tool_request_once(init_payload=init_payload, request=req)
    assert response.answer == "ok"


def test_run_tool_request_forwards_flow_id_to_ask_agent() -> None:
    seen: dict[str, object] = {}

    class _FakeAskAgent:
        def run(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(
                answer="ok",
                sources=[],
                mode="agent-tools",
                trace=[SimpleNamespace(to_dict=lambda: {"tool_name": "read_file", "status": "ok"})],
                warnings=[],
            )

    response = twp._run_tool_request(
        ask_agent=_FakeAskAgent(),  # type: ignore[arg-type]
        req=twp.ToolRunRequest(question="q", index_dir="/tmp/.mana/index", flow_id="flow-worker-1"),
        tools_only_strict_default=False,
        callbacks=None,
    )
    assert response.answer == "ok"
    assert seen["flow_id"] == "flow-worker-1"
