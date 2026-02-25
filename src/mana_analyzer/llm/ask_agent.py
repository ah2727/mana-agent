from __future__ import annotations

import json
import logging
from pathlib import Path
import shlex
import subprocess
from time import perf_counter

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from typing import Any, Sequence
from langchain_core.callbacks.base import BaseCallbackHandler
from mana_analyzer.analysis.models import AskResponseWithTrace, SearchHit, ToolInvocationTrace
from mana_analyzer.llm.prompts import ASK_AGENT_SYSTEM_PROMPT
from mana_analyzer.llm.run_logger import LlmRunLogger
from mana_analyzer.services.search_service import SearchService

logger = logging.getLogger(__name__)


class _SemanticSearchInput(BaseModel):
    query: str = Field(description="Query used for semantic code search")
    k: int = Field(default=8, ge=1, le=50, description="Top results to return")


class _ReadFileInput(BaseModel):
    path: str = Field(description="Absolute or project-relative file path")
    start_line: int = Field(default=1, ge=1)
    end_line: int = Field(default=200, ge=1)


class _RunCommandInput(BaseModel):
    cmd: str = Field(description="Shell command to execute in project root")


class AskAgent:
    _BLOCKED_PATTERNS = [
        "rm ",
        "mv ",
        "git reset --hard",
        "git checkout --",
        "sudo ",
        "dd ",
        "mkfs",
        "shutdown",
        "reboot",
        "chmod ",
        "chown ",
        ">",
        ">>",
    ]

    def __init__(
        self,
        api_key: str,
        model: str,
        search_service: SearchService,
        project_root: str | Path,
        base_url: str | None = None,
    ) -> None:
        kwargs = {"api_key": api_key, "model": model}
        if base_url:
            kwargs["base_url"] = base_url
        self.llm = ChatOpenAI(**kwargs)
        self.model = model
        self.search_service = search_service
        self.project_root = Path(project_root).resolve()
        self._resolved_index = self.project_root / ".mana_index"
        self._resolved_indexes = [self._resolved_index]
        self.run_logger = LlmRunLogger()

    def _is_blocked_command(self, cmd: str) -> bool:
        lowered = f"{cmd.lower()} "
        return any(pattern in lowered for pattern in self._BLOCKED_PATTERNS)

    def _build_tools(
        self, k_default: int, timeout_seconds: int
    ) -> tuple[list[StructuredTool], list[ToolInvocationTrace], list[SearchHit], list[str]]:
        traces: list[ToolInvocationTrace] = []
        sources: list[SearchHit] = []
        warnings: list[str] = []

        def semantic_search(query: str, k: int = k_default) -> str:
            started = perf_counter()
            status = "ok"
            output_preview = ""
            args_summary = f"query={query!r} k={k}"
            try:
                payload: list[dict] = []
                all_hits: list[SearchHit] = []
                for index_dir in self._resolved_indexes:
                    try:
                        hits = self.search_service.search(index_dir=index_dir, query=query, k=k)
                    except Exception as exc:
                        warning = f"Skipped unusable index {index_dir}: {exc}"
                        warnings.append(warning)
                        payload.append({"index_dir": str(index_dir), "error": str(exc)})
                        continue
                    all_hits.extend(hits)
                    payload.extend([{"index_dir": str(index_dir), **item.to_dict()} for item in hits])
                sources.extend(all_hits)
                encoded = json.dumps(payload)
                output_preview = encoded
                return encoded
            except Exception as exc:
                status = "error"
                output_preview = str(exc)
                return json.dumps({"error": str(exc)})
            finally:
                traces.append(
                    ToolInvocationTrace(
                        tool_name="semantic_search",
                        args_summary=args_summary,
                        duration_ms=(perf_counter() - started) * 1000,
                        status=status,
                        output_preview=output_preview,
                    )
                )

        def read_file(path: str, start_line: int = 1, end_line: int = 200) -> str:
            started = perf_counter()
            status = "ok"
            output_preview = ""
            args_summary = f"path={path!r} start={start_line} end={end_line}"
            try:
                requested = Path(path)
                resolved = requested if requested.is_absolute() else (self.project_root / requested)
                resolved = resolved.resolve()
                if self.project_root not in resolved.parents and resolved != self.project_root:
                    raise ValueError("path is outside project root")
                if not resolved.exists():
                    raise FileNotFoundError(str(resolved))
                lines = resolved.read_text(encoding="utf-8").splitlines()
                start = max(start_line, 1)
                end = max(end_line, start)
                end = min(end, start + 400)
                segment = lines[start - 1 : end]
                result = {
                    "file_path": str(resolved),
                    "start_line": start,
                    "end_line": min(end, len(lines)),
                    "content": "\n".join(segment),
                }
                encoded = json.dumps(result)
                output_preview = encoded
                return encoded
            except Exception as exc:
                status = "error"
                output_preview = str(exc)
                return json.dumps({"error": str(exc)})
            finally:
                traces.append(
                    ToolInvocationTrace(
                        tool_name="read_file",
                        args_summary=args_summary,
                        duration_ms=(perf_counter() - started) * 1000,
                        status=status,
                        output_preview=output_preview,
                    )
                )

        def run_command(cmd: str) -> str:
            started = perf_counter()
            status = "ok"
            output_preview = ""
            args_summary = f"cmd={cmd!r}"
            try:
                if self._is_blocked_command(cmd):
                    raise PermissionError("command blocked by safety policy")
                # Parse to fail fast on malformed shell syntax.
                shlex.split(cmd)
                completed = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    shell=True,
                    check=False,
                    timeout=timeout_seconds,
                    capture_output=True,
                    text=True,
                )
                payload = {
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[:4000],
                    "stderr": completed.stderr[:4000],
                }
                encoded = json.dumps(payload)
                output_preview = json.dumps(
                    {
                        "returncode": completed.returncode,
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                    }
                )
                return encoded
            except subprocess.TimeoutExpired:
                status = "timeout"
                output_preview = "command timed out"
                return json.dumps({"error": f"command timed out after {timeout_seconds}s"})
            except Exception as exc:
                status = "error"
                output_preview = str(exc)[:400]
                return json.dumps({"error": str(exc)})
            finally:
                traces.append(
                    ToolInvocationTrace(
                        tool_name="run_command",
                        args_summary=args_summary,
                        duration_ms=(perf_counter() - started) * 1000,
                        status=status,
                        output_preview=output_preview,
                    )
                )

        tools = [
            StructuredTool.from_function(
                func=semantic_search,
                name="semantic_search",
                description="Search indexed code semantically and return JSON list of hits.",
                args_schema=_SemanticSearchInput,
            ),
            StructuredTool.from_function(
                func=read_file,
                name="read_file",
                description="Read a file snippet and return JSON with file path and content.",
                args_schema=_ReadFileInput,
            ),
            StructuredTool.from_function(
                func=run_command,
                name="run_command",
                description="Run a non-destructive shell command in project root and return JSON stdout/stderr.",
                args_schema=_RunCommandInput,
            ),
        ]
        return tools, traces, sources, warnings
    def run(
        self,
        question: str,
        index_dir: str | Path,
        k: int,
        max_steps: int = 6,
        timeout_seconds: int = 30,
        index_dirs: list[str | Path] | None = None,
        callbacks: Sequence[BaseCallbackHandler] | None = None,
    ) -> AskResponseWithTrace:
        started = perf_counter()

        self._resolved_index = Path(index_dir).resolve()
        if index_dirs:
            self._resolved_indexes = sorted({Path(item).resolve() for item in index_dirs}, key=lambda item: str(item))
        else:
            self._resolved_indexes = [self._resolved_index]

        tools, traces, sources, warnings = self._build_tools(k_default=k, timeout_seconds=timeout_seconds)
        tool_map = {tool.name: tool for tool in tools}

        # Bind tools to LLM (still need to pass callbacks at invoke-time via config)
        bound = self.llm.bind_tools(tools)

        messages = [
            SystemMessage(content=ASK_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]

        # LangChain config (where callbacks are consumed)
        cfg: dict[str, Any] = {"callbacks": list(callbacks) if callbacks else []}

        final_answer = ""
        for _ in range(max_steps):
            # ✅ IMPORTANT: pass callbacks here so you get live updates (e.g. spinner)
            ai_msg = bound.invoke(messages, config=cfg)
            messages.append(ai_msg)

            tool_calls = getattr(ai_msg, "tool_calls", None) or []
            if not tool_calls:
                final_answer = str(ai_msg.content)
                break

            for call in tool_calls:
                name = str(call.get("name", ""))
                args = call.get("args", {}) or {}

                if name not in tool_map:
                    content = json.dumps({"error": f"unknown tool: {name}"})
                else:
                    try:
                        # ✅ IMPORTANT: pass callbacks to tool invoke so on_tool_start/on_tool_end fire
                        content = tool_map[name].invoke(args, config=cfg)
                    except Exception as exc:
                        content = json.dumps({"error": str(exc)})

                messages.append(ToolMessage(content=content, tool_call_id=str(call.get("id", ""))))

        if not final_answer:
            final_answer = "Tool loop reached the step limit before a final answer."

        deduped_sources = sorted(
            {
                (item.file_path, item.start_line, item.end_line, item.symbol_name): item
                for item in sources
            }.values(),
            key=lambda item: (item.file_path, item.start_line, item.end_line, item.symbol_name),
        )

        result = AskResponseWithTrace(
            answer=final_answer,
            sources=deduped_sources,
            mode="agent-tools",
            trace=traces,
            warnings=warnings,
        )

        run_logger = getattr(self, "run_logger", None)
        if run_logger is not None:
            run_logger.log(
                {
                    "flow": "ask-agent",
                    "model": getattr(self, "model", "unknown"),
                    "question_chars": len(question),
                    "question": question,
                    "index_dir": str(self._resolved_index),
                    "index_dirs": [str(item) for item in self._resolved_indexes],
                    "k": k,
                    "max_steps": max_steps,
                    "timeout_seconds": timeout_seconds,
                    "tool_calls": len(traces),
                    "trace": [item.to_dict() for item in traces],
                    "sources_count": len(result.sources),
                    "sources": [item.to_dict() for item in result.sources],
                    "duration_ms": round((perf_counter() - started) * 1000, 3),
                    "answer": result.answer,
                }
            )

        return result
    
    def run_multi(
        self,
        question: str,
        index_dirs: list[str | Path],
        k: int,
        max_steps: int = 6,
        timeout_seconds: int = 30,
    ) -> AskResponseWithTrace:
        resolved = sorted({Path(item).resolve() for item in index_dirs}, key=lambda item: str(item))
        if not resolved:
            return AskResponseWithTrace(
                answer="No indexes were provided for tool-enabled ask mode.",
                sources=[],
                mode="agent-tools",
                trace=[],
                warnings=["No index directories available."],
            )
        self._resolved_index = resolved[0]
        self._resolved_indexes = resolved
        return self.run(
            question=question,
            index_dir=self._resolved_index,
            index_dirs=resolved,
            k=k,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
        )
