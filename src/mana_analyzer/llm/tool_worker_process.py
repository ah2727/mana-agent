from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Literal

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.callbacks.base import BaseCallbackHandler
from pydantic import BaseModel, Field, ValidationError

from mana_analyzer.llm.ask_agent import AskAgent
from mana_analyzer.services.search_service import SearchService
from mana_analyzer.tools import (
    build_apply_patch_tool,
    build_search_internet_tool,
    build_write_file_tool,
)
from mana_analyzer.vector_store.faiss_store import FaissStore

logger = logging.getLogger(__name__)


class WorkerInitPayload(BaseModel):
    api_key: str
    model: str
    embed_model: str = "text-embedding-3-small"
    base_url: str | None = None
    project_root: str
    repo_root: str
    allowed_prefixes: list[str] | None = None
    tools_only_strict: bool = True


class ToolRunRequest(BaseModel):
    question: str
    index_dir: str | None = None
    index_dirs: list[str] | None = None
    k: int = 8
    max_steps: int = 6
    timeout_seconds: int = 30
    tool_policy: dict[str, Any] | None = None
    system_prompt: str | None = None
    tools_only_strict_override: bool | None = None


class ToolRunResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    mode: str = "agent-tools"
    trace: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class WorkerError(BaseModel):
    code: str
    message: str
    retriable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class WorkerEvent(BaseModel):
    name: str
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class WorkerEnvelope(BaseModel):
    type: Literal["init", "run_tools", "health", "shutdown"]
    request_id: str
    payload: dict[str, Any] = Field(default_factory=dict)


class WorkerReply(BaseModel):
    type: Literal["ok", "error", "event"]
    request_id: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ToolWorkerProcessError(RuntimeError):
    def __init__(self, *, code: str, message: str, retriable: bool = False, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.retriable = retriable
        self.details = details or {}


class ToolWorkerClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        repo_root: Path,
        project_root: Path,
        base_url: str | None = None,
        embed_model: str = "text-embedding-3-small",
        allowed_prefixes: list[str] | None = None,
        tools_only_strict: bool = True,
    ) -> None:
        self._init_payload = WorkerInitPayload(
            api_key=api_key,
            model=model,
            embed_model=embed_model,
            base_url=base_url,
            project_root=str(project_root.resolve()),
            repo_root=str(repo_root.resolve()),
            allowed_prefixes=allowed_prefixes,
            tools_only_strict=tools_only_strict,
        )
        self._proc: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "mana_analyzer.llm.tool_worker_process"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._request("init", self._init_payload.model_dump(), expect_event=True)

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.poll() is None:
            try:
                self._request_with_proc(proc, "shutdown", {}, expect_event=False)
            except Exception:
                pass
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()

    def health(self) -> dict[str, Any]:
        self.start()
        return self._request("health", {}, expect_event=False)

    def run_tools(
        self,
        request: ToolRunRequest,
        *,
        on_event: Callable[[WorkerEvent], None] | None = None,
    ) -> ToolRunResponse:
        self.start()
        try:
            payload = self._request("run_tools", request.model_dump(), expect_event=False, on_event=on_event)
        except ToolWorkerProcessError as exc:
            if self._can_retry(exc):
                self._restart()
                payload = self._request("run_tools", request.model_dump(), expect_event=False, on_event=on_event)
            else:
                raise
        return ToolRunResponse.model_validate(payload)

    def _can_retry(self, exc: ToolWorkerProcessError) -> bool:
        if exc.code == "tools_only_violation":
            return False
        if exc.retriable:
            return True
        return exc.code in {"worker_io_error", "worker_dead"}

    def _restart(self) -> None:
        self.stop()
        self.start()

    def _request(
        self,
        msg_type: str,
        payload: dict[str, Any],
        *,
        expect_event: bool,
        on_event: Callable[[WorkerEvent], None] | None = None,
    ) -> dict[str, Any]:
        proc = self._proc
        if proc is None:
            raise ToolWorkerProcessError(code="worker_dead", message="worker process is not running", retriable=True)
        return self._request_with_proc(proc, msg_type, payload, expect_event=expect_event, on_event=on_event)

    def _request_with_proc(
        self,
        proc: subprocess.Popen[str],
        msg_type: str,
        payload: dict[str, Any],
        *,
        expect_event: bool,
        on_event: Callable[[WorkerEvent], None] | None = None,
    ) -> dict[str, Any]:
        if proc.stdin is None or proc.stdout is None:
            raise ToolWorkerProcessError(code="worker_io_error", message="worker stdio unavailable", retriable=True)
        request_id = uuid.uuid4().hex
        envelope = WorkerEnvelope(type=msg_type, request_id=request_id, payload=payload)
        try:
            proc.stdin.write(envelope.model_dump_json() + "\n")
            proc.stdin.flush()
        except Exception as exc:
            raise ToolWorkerProcessError(
                code="worker_io_error",
                message=f"failed to write to worker: {exc}",
                retriable=True,
            ) from exc

        saw_event = False
        while True:
            line = proc.stdout.readline()
            if not line:
                raise ToolWorkerProcessError(code="worker_dead", message="worker terminated unexpectedly", retriable=True)
            try:
                reply = WorkerReply.model_validate_json(line.strip())
            except ValidationError as exc:
                raise ToolWorkerProcessError(
                    code="worker_protocol_error",
                    message=f"invalid worker reply: {exc}",
                    retriable=True,
                ) from exc
            if reply.request_id != request_id:
                continue
            if reply.type == "event":
                saw_event = True
                if on_event is not None:
                    try:
                        on_event(WorkerEvent.model_validate(reply.payload))
                    except Exception:
                        logger.debug("Failed to process worker event", exc_info=True)
                continue
            if reply.type == "error":
                err = WorkerError.model_validate(reply.payload)
                raise ToolWorkerProcessError(
                    code=err.code,
                    message=err.message,
                    retriable=err.retriable,
                    details=err.details,
                )
            if expect_event and not saw_event:
                raise ToolWorkerProcessError(
                    code="worker_protocol_error",
                    message="expected event before init confirmation",
                    retriable=True,
                )
            return reply.payload


class _WorkerToolEventCallback(BaseCallbackHandler):
    """Emit per-tool events from worker process back to parent client."""

    def __init__(self, *, request_id: str, emit_reply: Callable[[WorkerReply], None]) -> None:
        self._request_id = request_id
        self._emit_reply = emit_reply
        self._tool: str | None = None
        self._t0: float = 0.0

    def _emit(self, *, name: str, message: str, data: dict[str, Any] | None = None) -> None:
        self._emit_reply(
            WorkerReply(
                type="event",
                request_id=self._request_id,
                payload=WorkerEvent(name=name, message=message, data=data or {}).model_dump(),
            )
        )

    def on_tool_start(self, serialized, input_str: str, **kwargs) -> None:  # type: ignore[override]
        _ = kwargs
        tool = str((serialized or {}).get("name") or "tool")
        self._tool = tool
        self._t0 = time.time()
        args = (input_str or "").strip().replace("\n", " ")
        if len(args) > 160:
            args = args[:160] + "…"
        msg = f"TOOL start: {tool}"
        if args:
            msg += f" | args: {args}"
        self._emit(name="tool_start", message=msg, data={"tool": tool, "args": args})

    def on_tool_end(self, output: str, **kwargs) -> None:  # type: ignore[override]
        _ = (output, kwargs)
        tool = self._tool or "tool"
        dt = max(0.0, time.time() - self._t0)
        self._tool = None
        self._emit(
            name="tool_end",
            message=f"TOOL end: {tool} ({dt:0.1f}s)",
            data={"tool": tool, "duration_seconds": round(dt, 3)},
        )

    def on_tool_error(self, error: BaseException, **kwargs) -> None:  # type: ignore[override]
        _ = kwargs
        tool = self._tool or "tool"
        self._tool = None
        err = str(error).strip()
        msg = f"TOOL error: {tool}" + (f" - {err}" if err else "")
        self._emit(name="tool_error", message=msg, data={"tool": tool, "error": err})


class _ToolWorkerServer:
    def __init__(self) -> None:
        self._ask_agent: AskAgent | None = None
        self._tools_only_strict = True

    def run(self) -> int:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            try:
                env = WorkerEnvelope.model_validate_json(line)
            except ValidationError as exc:
                self._emit(
                    WorkerReply(
                        type="error",
                        request_id="unknown",
                        payload=WorkerError(
                            code="invalid_request",
                            message=f"request validation failed: {exc}",
                            retriable=False,
                        ).model_dump(),
                    )
                )
                continue
            if env.type == "init":
                self._handle_init(env)
                continue
            if env.type == "health":
                self._emit(WorkerReply(type="ok", request_id=env.request_id, payload={"status": "ok"}))
                continue
            if env.type == "shutdown":
                self._emit(WorkerReply(type="ok", request_id=env.request_id, payload={"status": "bye"}))
                return 0
            if env.type == "run_tools":
                self._handle_run_tools(env)
                continue
        return 0

    def _handle_init(self, env: WorkerEnvelope) -> None:
        try:
            payload = WorkerInitPayload.model_validate(env.payload)
            embeddings = OpenAIEmbeddings(
                api_key=payload.api_key,
                model=payload.embed_model,
                base_url=payload.base_url,
            )
            search_service = SearchService(store=FaissStore(embeddings))
            ask_agent = AskAgent(
                api_key=payload.api_key,
                model=payload.model,
                base_url=payload.base_url,
                search_service=search_service,
                project_root=Path(payload.project_root),
            )
            ask_agent.tools.extend(
                [
                    build_write_file_tool(
                        repo_root=Path(payload.repo_root),
                        allowed_prefixes=tuple(payload.allowed_prefixes) if payload.allowed_prefixes else None,
                    ),
                    build_apply_patch_tool(
                        repo_root=Path(payload.repo_root),
                        allowed_prefixes=tuple(payload.allowed_prefixes) if payload.allowed_prefixes else None,
                    ),
                    build_search_internet_tool(),
                ]
            )
            self._ask_agent = ask_agent
            self._tools_only_strict = bool(payload.tools_only_strict)
            self._emit(
                WorkerReply(
                    type="event",
                    request_id=env.request_id,
                    payload=WorkerEvent(name="initialized", message="worker initialized").model_dump(),
                )
            )
            self._emit(WorkerReply(type="ok", request_id=env.request_id, payload={"status": "ok"}))
        except Exception as exc:
            self._emit(
                WorkerReply(
                    type="error",
                    request_id=env.request_id,
                    payload=WorkerError(
                        code="init_failed",
                        message=str(exc),
                        retriable=False,
                    ).model_dump(),
                )
            )

    def _handle_run_tools(self, env: WorkerEnvelope) -> None:
        if self._ask_agent is None:
            self._emit(
                WorkerReply(
                    type="error",
                    request_id=env.request_id,
                    payload=WorkerError(
                        code="not_initialized",
                        message="worker is not initialized",
                        retriable=True,
                    ).model_dump(),
                )
            )
            return
        try:
            req = ToolRunRequest.model_validate(env.payload)
            tool_event_cb = _WorkerToolEventCallback(request_id=env.request_id, emit_reply=self._emit)
            if req.index_dirs:
                result = self._ask_agent.run_multi(
                    question=req.question,
                    index_dirs=[Path(p) for p in req.index_dirs],
                    k=req.k,
                    max_steps=req.max_steps,
                    timeout_seconds=req.timeout_seconds,
                    system_prompt=req.system_prompt,
                    tool_policy=req.tool_policy,
                    callbacks=[tool_event_cb],
                )
            else:
                if not req.index_dir:
                    raise ValueError("index_dir or index_dirs must be provided")
                result = self._ask_agent.run(
                    question=req.question,
                    index_dir=Path(req.index_dir),
                    k=req.k,
                    max_steps=req.max_steps,
                    timeout_seconds=req.timeout_seconds,
                    system_prompt=req.system_prompt,
                    tool_policy=req.tool_policy,
                    callbacks=[tool_event_cb],
                )
            trace_rows = [item.to_dict() for item in getattr(result, "trace", [])]
            ok_tools = len([row for row in trace_rows if str(row.get("status", "")).lower().strip() == "ok"])
            strict_required = self._tools_only_strict
            if req.tools_only_strict_override is not None:
                strict_required = bool(req.tools_only_strict_override)
            if strict_required and ok_tools <= 0:
                self._emit(
                    WorkerReply(
                        type="error",
                        request_id=env.request_id,
                        payload=WorkerError(
                            code="tools_only_violation",
                            message="tools-only mode requires at least one successful tool call",
                            retriable=False,
                            details={"trace_count": len(trace_rows)},
                        ).model_dump(),
                    )
                )
                return
            response = ToolRunResponse(
                answer=str(getattr(result, "answer", "")),
                sources=[item.to_dict() for item in getattr(result, "sources", [])],
                mode=str(getattr(result, "mode", "agent-tools")),
                trace=trace_rows,
                warnings=[str(item) for item in getattr(result, "warnings", [])],
            )
            self._emit(WorkerReply(type="ok", request_id=env.request_id, payload=response.model_dump()))
        except Exception as exc:
            self._emit(
                WorkerReply(
                    type="error",
                    request_id=env.request_id,
                    payload=WorkerError(
                        code="run_failed",
                        message=str(exc),
                        retriable=True,
                    ).model_dump(),
                )
            )

    @staticmethod
    def _emit(reply: WorkerReply) -> None:
        sys.stdout.write(reply.model_dump_json() + "\n")
        sys.stdout.flush()


def main() -> int:
    return _ToolWorkerServer().run()


if __name__ == "__main__":
    raise SystemExit(main())
