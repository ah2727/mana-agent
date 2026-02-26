from __future__ import annotations

import json
import logging
from pathlib import Path
import shlex
import subprocess
from time import perf_counter
from typing import Any, Sequence, Optional
from collections import defaultdict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool, BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

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

        # ✅ NEW: allow external code to register extra tools (e.g. write_file/apply_patch)
        self.tools: list[BaseTool] = []

    def _is_blocked_command(self, cmd: str) -> bool:
        lowered = f"{cmd.lower()} "
        return any(pattern in lowered for pattern in self._BLOCKED_PATTERNS)

    def _build_tools(
        self, k_default: int, timeout_seconds: int
    ) -> tuple[list[BaseTool], list[ToolInvocationTrace], list[SearchHit], list[str]]:
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

        base_tools: list[BaseTool] = [
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

        # ✅ NEW: include any externally-registered tools (write_file/apply_patch/etc)
        all_tools = [*base_tools, *list(self.tools or [])]

        return all_tools, traces, sources, warnings

    # ✅ NEW: public "ask" API (what your CodingAgent expects)
    def ask(
        self,
        question: str,
        *,
        index_dir: str | Path,
        k: int,
        max_steps: int = 9999999999999,
        timeout_seconds: int = 9999999999999,
        index_dirs: list[str | Path] | None = None,
        callbacks: Sequence[BaseCallbackHandler] | None = None,
        system_prompt: str | None = None,
        tool_use: bool = True,
    ) -> AskResponseWithTrace:
        # tool_use kept for compatibility; this AskAgent is tool-based by design.
        return self.run(
            question=question,
            index_dir=index_dir,
            k=k,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
            index_dirs=index_dirs,
            callbacks=callbacks,
            system_prompt=system_prompt,
        )

    def run(
        self,
        question: str,
        index_dir: str | Path,
        k: int,
        max_steps: int = 6,
        timeout_seconds: int = 30,
        index_dirs: list[str | Path] | None = None,
        callbacks: Sequence[BaseCallbackHandler] | None = None,
        system_prompt: str | None = None,  # ✅ NEW
    ) -> AskResponseWithTrace:
        started = perf_counter()

        self._resolved_index = Path(index_dir).resolve()
        if index_dirs:
            self._resolved_indexes = sorted({Path(item).resolve() for item in index_dirs}, key=lambda item: str(item))
        else:
            self._resolved_indexes = [self._resolved_index]

        tools, traces, sources, warnings = self._build_tools(k_default=k, timeout_seconds=timeout_seconds)
        tool_map = {tool.name: tool for tool in tools}

        bound = self.llm.bind_tools(tools)

        messages = [
            SystemMessage(content=system_prompt or ASK_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]

        cfg: dict[str, Any] = {"callbacks": list(callbacks) if callbacks else []}

        final_answer = ""
        for _ in range(max_steps):
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
                        content = tool_map[name].invoke(args, config=cfg)
                    except Exception as exc:
                        content = json.dumps({"error": str(exc)})

                messages.append(ToolMessage(content=content, tool_call_id=str(call.get("id", ""))))

        if not final_answer:
            final_answer = "Tool loop reached the step limit before a final answer."

        deduped_sources = sorted(
            {(item.file_path, item.start_line, item.end_line, item.symbol_name): item for item in sources}.values(),
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
        *,
        question: str,
        index_dirs: Sequence[str | Path],
        k: int,
        max_steps: int = 6,
        timeout_seconds: int = 30,
        callbacks: Sequence[Any] | None = None,
        **kwargs: Any,
    ):
        """
        Dir-mode entrypoint.

        This implementation:
        1) Searches across multiple indexes (SearchService.search_multi).
        2) Chooses the best single index to run the agent loop against (highest top-hit score).
        3) Calls self.run(...) using that chosen index_dir.
        4) Returns the agent result, but keeps the multi-index sources + warnings.
        """
        if getattr(self, "search_service", None) is None:
            raise RuntimeError("AskAgent.search_service is required for run_multi()")

        resolved_indexes = [Path(p).resolve() for p in index_dirs]
        if not resolved_indexes:
            raise RuntimeError("run_multi(): index_dirs is empty")

        # 1) Retrieve across all indexes
        sources, warnings = self.search_service.search_multi(  # type: ignore[attr-defined]
            index_dirs=resolved_indexes,
            query=question,
            k=k,
        )

        if not sources:
            # If your project has AskResponseWithTrace model, return it.
            # Otherwise return a simple dict-like fallback.
            try:
                from mana_analyzer.analysis.models import AskResponseWithTrace  # type: ignore

                return AskResponseWithTrace(
                    answer="No relevant indexed code context found across indexes.",
                    sources=[],
                    source_groups=[],
                    warnings=warnings or [],
                    mode="agent-tools",
                    trace=[],
                )
            except Exception:
                return {
                    "answer": "No relevant indexed code context found across indexes.",
                    "sources": [],
                    "warnings": warnings or [],
                    "mode": "agent-tools",
                    "trace": [],
                }

        # 2) Choose best index based on highest top-hit score per index bucket
        # We infer which index a hit belongs to by matching hit.file_path under index_dir.parent
        best_index = resolved_indexes[0]
        best_score = float("-inf")

        # score_by_index: index_dir -> max_score
        score_by_index: dict[Path, float] = defaultdict(lambda: float("-inf"))
        for hit in sources:
            try:
                hit_path = Path(hit.file_path).resolve()
            except Exception:
                continue

            for idx in resolved_indexes:
                subproject_root = idx.parent
                if hit_path == subproject_root or subproject_root in hit_path.parents:
                    try:
                        score = float(getattr(hit, "score", 0.0))
                    except Exception:
                        score = 0.0
                    if score > score_by_index[idx]:
                        score_by_index[idx] = score

        for idx, sc in score_by_index.items():
            if sc > best_score:
                best_score = sc
                best_index = idx

        # 3) Run the normal agent loop against the chosen index
        result = self.run(
            question=question,
            index_dir=best_index,
            k=k,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
            callbacks=callbacks,
            **kwargs,
        )

        # 4) Attach multi-index sources + warnings so CLI can show them consistently
        # (works for AskResponseWithTrace dataclass-like objects)
        try:
            if hasattr(result, "sources"):
                result.sources = sources
            if hasattr(result, "warnings"):
                result.warnings = list(getattr(result, "warnings", []) or [])
                if warnings:
                    result.warnings.extend(warnings)
                result.warnings.append(f"dir-mode: selected index for tool-run = {best_index}")
            if hasattr(result, "mode") and getattr(result, "mode", None):
                # keep existing mode
                pass
            elif hasattr(result, "mode"):
                result.mode = "agent-tools-dir"
        except Exception:
            # If it's a string/dict/etc, just return as-is
            pass

        return result