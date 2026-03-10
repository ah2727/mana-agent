from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

# افزودن کتابخانه‌های مورد نیاز برای Retry
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from langchain_openai import OpenAIEmbeddings
from langchain_core.callbacks.base import BaseCallbackHandler
from pydantic import BaseModel, Field, ValidationError

from mana_analyzer.services.coding_memory_service import CodingMemoryService
from mana_analyzer.services.search_service import SearchService
from mana_analyzer.llm.ask_agent import AskAgent
from mana_analyzer.tools import build_apply_patch_tool, build_write_file_tool
from mana_analyzer.tools import safe_apply_patch, safe_write_file
from mana_analyzer.tools.search_internet import build_search_internet_tool, safe_search_internet
from mana_analyzer.vector_store.faiss_store import FaissStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keys that must NEVER be sent to provider APIs as message/input parameters.
# Some providers (e.g. certain OpenAI-compatible proxies) reject unknown keys
# such as "status" when they appear inside the `input` / `messages` array.
# ---------------------------------------------------------------------------
_PROVIDER_BANNED_KEYS = frozenset({"status"})


def _strip_banned_keys(obj: Any) -> Any:
    """Recursively remove keys that are not accepted by the provider API.

    Works on dicts, lists, and nested combinations thereof.  Returns a
    *new* object — the original is never mutated.
    """
    if isinstance(obj, dict):
        return {
            k: _strip_banned_keys(v)
            for k, v in obj.items()
            if k not in _PROVIDER_BANNED_KEYS
        }
    if isinstance(obj, list):
        return [_strip_banned_keys(item) for item in obj]
    return obj


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
    flow_id: str | None = None
    k: int = 8
    max_steps: int = 6
    timeout_seconds: int = 30
    tool_policy: dict[str, Any] | None = None
    system_prompt: str | None = None
    tools_only_strict_override: bool | None = None
    tool_name: str = ""
    tool_args: dict[str, Any] = Field(default_factory=dict)


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
    type: Literal["init", "run_tools", "health", "shutdown", "update_model"]
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

    def init_payload_dict(self) -> dict[str, Any]:
        return self._init_payload.model_dump()

    def update_model(self, model_name: str) -> None:
        """Update the model name and sync with the worker process."""
        self._init_payload.model = model_name
        if self._proc and self._proc.poll() is None:
            try:
                # سعی برای آپدیت درجا
                self._request("update_model", {"model": model_name}, expect_event=False)
                logger.info(f"Worker model updated to {model_name} in-place.")
            except Exception:
                logger.warning("Failed to update worker model in-place, restarting worker.")
                self._restart()

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        env = os.environ.copy()
        repo_root = Path(self._init_payload.repo_root).resolve()
        pythonpath_entries: list[str] = []
        src_dir = repo_root / "src"
        if src_dir.exists():
            pythonpath_entries.append(str(src_dir))
        pythonpath_entries.append(str(repo_root))
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            pythonpath_entries.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "mana_analyzer.llm.tool_worker_process"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env,
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ToolWorkerProcessError, IOError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def run_tools(
        self,
        request: ToolRunRequest,
        *,
        on_event: Callable[[WorkerEvent], None] | None = None,
    ) -> ToolRunResponse:
        self.start()

        # Prepare the payload safely — strip all provider-banned keys
        # so that "status" never leaks into the API request body.
        payload_dict = _strip_banned_keys(request.model_dump())

        # Double-check: also sanitize tool_args explicitly
        if "tool_args" in payload_dict:
            payload_dict["tool_args"] = self._prepare_tool_input(payload_dict["tool_args"])

        try:
            response_payload = self._request(
                "run_tools",
                payload_dict,
                expect_event=False,
                on_event=on_event,
            )
        except ToolWorkerProcessError as exc:
            if self._is_status_param_error(exc):
                logger.warning(
                    "Provider rejected 'status' parameter — stripping and retrying. "
                    "Original error: %s",
                    exc,
                )
                self._restart()
                raise  # let @retry handle it after restart

            if self._can_retry(exc):
                logger.warning(
                    "Retrying run_tools due to worker error: %s. Message: %s",
                    exc.code,
                    exc,
                )
                self._restart()
                raise
            else:
                raise

        return ToolRunResponse.model_validate(response_payload)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_status_param_error(exc: ToolWorkerProcessError) -> bool:
        """Detect the specific 'Unknown parameter … status' provider error."""
        msg = str(exc).lower()
        return "unknown parameter" in msg and "status" in msg

    def _prepare_tool_input(self, tool_args: dict[str, Any]) -> dict[str, Any]:
        """
        Prepare the tool input for the provider.
        Strips every key that the provider will reject (e.g. ``status``).
        """
        return _strip_banned_keys(tool_args)

    def _can_retry(self, exc: ToolWorkerProcessError) -> bool:
        if exc.code == "tools_only_violation":
            return False
        if exc.retriable:
            return True
        # اگر مدل پیدا نشد یا خطای IO بود، اجازه ریترای بده
        return exc.code in {"worker_io_error", "worker_dead", "model_not_found", "init_failed"}

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

        # ── Sanitize the outbound payload so banned keys never reach the wire ──
        sanitized_payload = _strip_banned_keys(payload)

        envelope = WorkerEnvelope(type=msg_type, request_id=request_id, payload=sanitized_payload)
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


def _build_worker_ask_agent(payload: WorkerInitPayload) -> AskAgent:
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
        coding_memory_service=CodingMemoryService(project_root=Path(payload.project_root)),
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
    return ask_agent


def _sanitize_trace_for_provider(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of *trace_rows* with provider-banned keys removed.

    The ``status`` field is perfectly fine for *internal* bookkeeping (we
    still use it for the ``tools_only_strict`` check), but it must never
    be forwarded to the upstream LLM provider because some providers
    (e.g. certain OpenAI-compatible proxies) reject it with::

        Unknown parameter: 'input[N].status'

    This helper creates a sanitised copy that is safe to embed in any
    payload that will eventually be serialised and sent over the wire.
    """
    return _strip_banned_keys(trace_rows)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _run_tool_request(
    *,
    ask_agent: AskAgent,
    req: ToolRunRequest,
    tools_only_strict_default: bool,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> ToolRunResponse:
    if req.index_dirs:
        result = ask_agent.run_multi(
            question=req.question,
            index_dirs=[Path(p) for p in req.index_dirs],
            flow_id=req.flow_id,
            k=req.k,
            max_steps=req.max_steps,
            timeout_seconds=req.timeout_seconds,
            system_prompt=req.system_prompt,
            tool_policy=req.tool_policy,
            callbacks=callbacks,
        )
    else:
        if not req.index_dir:
            raise ValueError("index_dir or index_dirs must be provided")
        result = ask_agent.run(
            question=req.question,
            index_dir=Path(req.index_dir),
            flow_id=req.flow_id,
            k=req.k,
            max_steps=req.max_steps,
            timeout_seconds=req.timeout_seconds,
            system_prompt=req.system_prompt,
            tool_policy=req.tool_policy,
            callbacks=callbacks,
        )

    # Build the raw trace — still contains "status" for local logic.
    trace_rows_raw = [item.to_dict() for item in getattr(result, "trace", [])]

    # Use the RAW trace (with status) for the strict-tools check.
    ok_tools = len(
        [row for row in trace_rows_raw if str(row.get("status", "")).lower().strip() == "ok"]
    )
    strict_required = bool(tools_only_strict_default)
    if req.tools_only_strict_override is not None:
        strict_required = bool(req.tools_only_strict_override)
    if strict_required and ok_tools <= 0:
        raise ToolWorkerProcessError(
            code="tools_only_violation",
            message="tools-only mode requires at least one successful tool call",
            retriable=False,
            details={"trace_count": len(trace_rows_raw)},
        )

    # Sanitize trace before it leaves the process — remove "status" and
    # any other keys the provider would reject.
    trace_rows_safe = _sanitize_trace_for_provider(trace_rows_raw)

    return ToolRunResponse(
        answer=str(getattr(result, "answer", "")),
        sources=[_strip_banned_keys(item.to_dict()) for item in getattr(result, "sources", [])],
        mode=str(getattr(result, "mode", "agent-tools")),
        trace=trace_rows_safe,
        warnings=[str(item) for item in getattr(result, "warnings", [])],
    )


def run_tool_request_once(
    *,
    init_payload: WorkerInitPayload,
    request: ToolRunRequest,
) -> ToolRunResponse:
    ask_agent = _build_worker_ask_agent(init_payload)
    return _run_tool_request(
        ask_agent=ask_agent,
        req=request,
        tools_only_strict_default=bool(init_payload.tools_only_strict),
        callbacks=None,
    )


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


@dataclass
class TurnToolExecutionState:
    executed_tools: set[str] = field(default_factory=set)

    def reset(self) -> None:
        self.executed_tools.clear()

    def claim(self, tool_name: str) -> bool:
        canonical = str(tool_name or "").strip().lower()
        if not canonical:
            return True
        if canonical in self.executed_tools:
            return False
        self.executed_tools.add(canonical)
        return True


class _ToolWorkerServer:
    def __init__(self) -> None:
        self._ask_agent: AskAgent | None = None
        self._tools_only_strict = True
        self._turn_tool_state = TurnToolExecutionState()

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
            if env.type == "update_model":
                self._handle_update_model(env)
                continue
            if env.type == "health":
                self._emit(WorkerReply(type="ok", request_id=env.request_id, payload={"healthy": True}))
                continue
            if env.type == "shutdown":
                self._emit(WorkerReply(type="ok", request_id=env.request_id, payload={"shutdown": True}))
                return 0
            if env.type == "run_tools":
                self._handle_run_tools(env)
                continue
        return 0

    def _handle_init(self, env: WorkerEnvelope) -> None:
        try:
            payload = WorkerInitPayload.model_validate(env.payload)
            self._ask_agent = _build_worker_ask_agent(payload)
            self._tools_only_strict = bool(payload.tools_only_strict)
            self._emit(
                WorkerReply(
                    type="event",
                    request_id=env.request_id,
                    payload=WorkerEvent(name="initialized", message="worker initialized").model_dump(),
                )
            )
            self._emit(WorkerReply(type="ok", request_id=env.request_id, payload={"initialized": True}))
        except Exception as exc:
            self._emit(
                WorkerReply(
                    type="error",
                    request_id=env.request_id,
                    payload=WorkerError(
                        code="init_failed",
                        message=str(exc),
                        retriable=True,  # اجازه ریترای برای خطاهای احتمالی شبکه در زمان اینیت
                    ).model_dump(),
                )
            )

    def _handle_update_model(self, env: WorkerEnvelope) -> None:
        """Dynamically update the model name of the active AskAgent."""
        if self._ask_agent is None:
            self._emit(WorkerReply(type="error", request_id=env.request_id,
                                   payload=WorkerError(code="not_initialized", message="worker not initialized").model_dump()))
            return

        new_model = env.payload.get("model")
        if new_model:
            try:
                self._ask_agent.model = new_model
                # اگر متد آپدیت در AskAgent پیاده‌سازی شده باشد
                if hasattr(self._ask_agent, "update_model"):
                    self._ask_agent.update_model(new_model)
                self._emit(WorkerReply(type="ok", request_id=env.request_id, payload={"updated": True, "model": new_model}))
            except Exception as exc:
                self._emit(WorkerReply(type="error", request_id=env.request_id,
                                       payload=WorkerError(code="update_failed", message=str(exc)).model_dump()))

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
            # ── Sanitize the inbound envelope so "status" never reaches the agent ──
            sanitized_env_payload = _strip_banned_keys(env.payload)
            req = ToolRunRequest.model_validate(sanitized_env_payload)

            tool_name = str(req.tool_name or "").strip()
            if tool_name:
                if not self._turn_tool_state.claim(tool_name):
                    logger.warning(
                        "Blocking duplicate tool execution within turn: tool=%s turn=%s",
                        tool_name,
                        env.request_id,
                    )
                    self._emit(
                        WorkerReply(
                            type="ok",
                            request_id=env.request_id,
                            payload=ToolRunResponse(
                                answer="Tool already executed in this turn.",
                                sources=[],
                                mode="agent-tools",
                                trace=[
                                    {
                                        "tool_name": tool_name,
                                        "result": "duplicate_blocked",
                                        "turn_id": env.request_id,
                                        "message": "Tool already executed in this turn.",
                                    }
                                ],
                                warnings=["Tool already executed in this turn."],
                            ).model_dump(),
                        )
                    )
                    return
                logger.info("Registered tool execution for turn: tool=%s turn=%s", tool_name, env.request_id)
            tool_event_cb = _WorkerToolEventCallback(request_id=env.request_id, emit_reply=self._emit)
            response = _run_tool_request(
                ask_agent=self._ask_agent,
                req=req,
                tools_only_strict_default=self._tools_only_strict,
                callbacks=[tool_event_cb],
            )

            # ── Final safety net: strip banned keys from the outbound payload ──
            safe_payload = _strip_banned_keys(response.model_dump())
            self._emit(WorkerReply(type="ok", request_id=env.request_id, payload=safe_payload))
        except ToolWorkerProcessError as exc:
            self._emit(
                WorkerReply(
                    type="error",
                    request_id=env.request_id,
                    payload=WorkerError(
                        code=exc.code,
                        message=str(exc),
                        retriable=bool(exc.retriable),
                        details=exc.details,
                    ).model_dump(),
                )
            )
        except Exception as exc:
            # تشخیص هوشمند خطای مدل نامعتبر (404)
            code = "run_failed"
            err_msg = str(exc).lower()
            if "model_not_found" in err_msg or "404" in err_msg or "not found" in err_msg:
                code = "model_not_found"

            # ── Detect the specific "Unknown parameter: status" provider error ──
            retriable = True
            if "unknown parameter" in err_msg and "status" in err_msg:
                code = "provider_unknown_param"
                logger.error(
                    "Provider rejected a banned parameter (likely 'status'). "
                    "This should have been stripped — please report this as a bug. "
                    "Original error: %s",
                    exc,
                )

            self._emit(
                WorkerReply(
                    type="error",
                    request_id=env.request_id,
                    payload=WorkerError(
                        code=code,
                        message=str(exc),
                        retriable=retriable,
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
