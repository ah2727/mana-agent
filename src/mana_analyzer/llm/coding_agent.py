"""
mana_analyzer.llm.coding_agent

A thin wrapper around the existing AskAgent that enables:
- safe code modifications (write_file, apply_patch)
- post-change verification via static analysis
- diff and change summarisation for human review

This version:
- supports base_url
- supports dir-mode without requiring AskAgent.run_multi
- supports "edit anywhere under repo_root" via allowed_prefixes=None
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

from mana_analyzer.tools import build_apply_patch_tool, build_write_file_tool

logger = logging.getLogger(__name__)

CODING_SYSTEM_PROMPT = """\
You are a coding agent operating inside a repository.

Rules:
- Prefer apply_patch (unified diff) for edits to existing files.
- Use write_file only for new files or when explicitly asked to overwrite.
- Only modify files under src/ and tests/ unless the user explicitly asks otherwise.
- After changes, aim for clean static checks; avoid unused imports and obvious style issues.
- When you create new public functions/classes, add docstrings and type hints.
"""


class AskAgentLike(Protocol):
    tools: list[Any]

    # Some repos expose ask(...), some expose run(...), some expose both.
    def ask(self, question: str, **kwargs: Any) -> Any:  # pragma: no cover
        ...


def _as_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_as_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _as_jsonable(v) for k, v in obj.items()}
    return obj


class CodingAgent:
    """
    Wraps an existing AskAgent (tool-capable) and adds mutation tools
    + post-run verification artifacts (git diff, static analysis, etc).

    IMPORTANT:
    - allowed_prefixes=None => allow writes anywhere under repo_root (still blocked from escaping repo_root)
    """

    def __init__(
        self,
        *,
        api_key: str,
        repo_root: Path,
        ask_agent: AskAgentLike,
        base_url: str | None = None,
        allowed_prefixes: Optional[Sequence[str]] = ("src/", "tests/"),
        system_prompt: str = CODING_SYSTEM_PROMPT,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.repo_root = repo_root.resolve()
        self.ask_agent: AskAgentLike = ask_agent
        self.allowed_prefixes = allowed_prefixes
        self.system_prompt = system_prompt

        # Attach safe mutation tools to the underlying agent.
        # allowed_prefixes=None -> unrestricted within repo_root
        self.ask_agent.tools.extend(
            [
                build_write_file_tool(repo_root=self.repo_root, allowed_prefixes=self.allowed_prefixes),
                build_apply_patch_tool(repo_root=self.repo_root, allowed_prefixes=self.allowed_prefixes),
            ]
        )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def generate(
        self,
        request: str,
        *,
        index_dir: str | Path | None = None,
        k: int | None = None,
        max_steps: int = 9999999999999999999,
        timeout_seconds: int = 9999999999999999999,
        callbacks: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        """
        Classic (single-index) coding request.
        - If the underlying agent supports `run(...)`, we use it.
        - Otherwise we fallback to `ask(...)`.
        """
        before = self._git_status_paths()

        answer = self._call_agent_single(
            request,
            index_dir=index_dir,
            k=k,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
            callbacks=callbacks,
        )

        after = self._git_status_paths()
        changed = sorted(after.difference(before))
        findings = self._run_static_analysis([p for p in changed if p.endswith(".py")])
        diff = self._git_diff(changed)

        status = "ok" if not findings else "warning"
        return {
            "status": status,
            "answer": answer,
            "changed_files": changed,
            "diff": diff,
            "static_analysis": {
                "finding_count": len(findings),
                "findings": [_as_jsonable(f) for f in findings],
            },
        }

    def generate_dir_mode(
        self,
        request: str,
        *,
        index_dirs: Sequence[str | Path],
        k: int | None = None,
        max_steps: int = 12,
        timeout_seconds: int = 60,
        callbacks: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        """
        Dir-mode coding request.

        Key point: DO NOT assume AskAgent.run_multi exists.
        We try:
          1) run_multi(...) if present
          2) run(...) if present (by embedding index list into the prompt)
          3) ask(...) fallback
        """
        before = self._git_status_paths()

        answer = self._call_agent_multi(
            request,
            index_dirs=index_dirs,
            k=k,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
            callbacks=callbacks,
        )

        after = self._git_status_paths()
        changed = sorted(after.difference(before))
        findings = self._run_static_analysis([p for p in changed if p.endswith(".py")])
        diff = self._git_diff(changed)

        status = "ok" if not findings else "warning"
        return {
            "status": status,
            "answer": answer,
            "changed_files": changed,
            "diff": diff,
            "static_analysis": {
                "finding_count": len(findings),
                "findings": [_as_jsonable(f) for f in findings],
            },
        }

    # ---------------------------------------------------------------------
    # Agent call helpers
    # ---------------------------------------------------------------------

    def _call_agent_single(
        self,
        request: str,
        *,
        index_dir: str | Path | None,
        k: int | None,
        max_steps: int,
        timeout_seconds: int,
        callbacks: Sequence[Any] | None,
    ) -> str:
        # Prefer an actual tool-loop runner if present
        if hasattr(self.ask_agent, "run"):
            return self._stringify(
                self._invoke_run_like(
                    "run",
                    question=request,
                    index_dir=str(Path(index_dir).resolve()) if index_dir is not None else None,
                    k=k,
                    max_steps=max_steps,
                    timeout_seconds=timeout_seconds,
                    callbacks=callbacks,
                )
            )

        # Fallback: ask(...)
        return self._call_ask_like(request)

    def _call_agent_multi(
        self,
        request: str,
        *,
        index_dirs: Sequence[str | Path],
        k: int | None,
        max_steps: int,
        timeout_seconds: int,
        callbacks: Sequence[Any] | None,
    ) -> str:
        resolved = [str(Path(p).resolve()) for p in index_dirs if str(p).strip()]
        if not resolved:
            return "No index_dirs provided for dir-mode."

        # 1) If run_multi exists, use it.
        if hasattr(self.ask_agent, "run_multi"):
            return self._stringify(
                self._invoke_run_like(
                    "run_multi",
                    question=request,
                    index_dirs=resolved,
                    k=k,
                    max_steps=max_steps,
                    timeout_seconds=timeout_seconds,
                    callbacks=callbacks,
                )
            )

        # 2) If only run exists, embed index selection in the prompt.
        if hasattr(self.ask_agent, "run"):
            stitched = (
                f"{request}\n\n"
                "DIR-MODE CONTEXT:\n"
                f"- index_dirs:\n  - " + "\n  - ".join(resolved) + "\n"
                "- Use these indexes when searching context.\n"
            )
            return self._stringify(
                self._invoke_run_like(
                    "run",
                    question=stitched,
                    index_dir=resolved[0],  # some agents require an index_dir param; give the first
                    k=k,
                    max_steps=max_steps,
                    timeout_seconds=timeout_seconds,
                    callbacks=callbacks,
                )
            )

        # 3) Fallback: ask(...)
        stitched = (
            f"{request}\n\n"
            "DIR-MODE CONTEXT:\n"
            f"- index_dirs:\n  - " + "\n  - ".join(resolved) + "\n"
            "- You must use tools to open files; do not guess.\n"
        )
        return self._call_ask_like(stitched)

    def _call_ask_like(self, request: str) -> str:
        kwargs: dict[str, Any] = {}
        try:
            sig = inspect.signature(self.ask_agent.ask)
            if "tool_use" in sig.parameters:
                kwargs["tool_use"] = True
            if "system_prompt" in sig.parameters:
                kwargs["system_prompt"] = self.system_prompt
            elif "instructions" in sig.parameters:
                kwargs["instructions"] = self.system_prompt
        except Exception:
            pass

        if "system_prompt" not in kwargs and "instructions" not in kwargs:
            request = f"{self.system_prompt}\n\nUser request:\n{request}"

        result = self.ask_agent.ask(request, **kwargs)
        return self._stringify(result)

    def _invoke_run_like(self, method_name: str, **args: Any) -> Any:
        """
        Call self.ask_agent.<method_name>(...) but only pass args that exist in the signature.
        This avoids TypeError when different agent implementations use different params.
        """
        fn = getattr(self.ask_agent, method_name)
        try:
            sig = inspect.signature(fn)
            filtered: dict[str, Any] = {}
            for k, v in args.items():
                if v is None:
                    continue
                if k in sig.parameters:
                    filtered[k] = v

            # Try to pass system prompt if supported
            if "system_prompt" in sig.parameters:
                filtered["system_prompt"] = self.system_prompt
            elif "instructions" in sig.parameters:
                filtered["instructions"] = self.system_prompt

            return fn(**filtered)
        except Exception:
            # last resort: call with just question if possible
            try:
                return fn(args.get("question"))
            except Exception:
                raise

    def _stringify(self, result: Any) -> str:
        if isinstance(result, str):
            return result
        try:
            return json.dumps(_as_jsonable(result), indent=2, ensure_ascii=False)
        except Exception:
            return str(result)

    # ---------------------------------------------------------------------
    # Verification helpers
    # ---------------------------------------------------------------------

    def _git_status_paths(self) -> set[str]:
        try:
            proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                return set()

            paths: set[str] = set()
            for line in proc.stdout.splitlines():
                if len(line) >= 4:
                    p = line[3:].strip()
                    if p:
                        paths.add(p.replace("\\", "/"))
            return paths
        except Exception:
            return set()

    def _git_diff(self, paths: list[str]) -> str:
        if not paths:
            return ""
        try:
            proc = subprocess.run(
                ["git", "diff", "--", *paths],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                return ""
            return proc.stdout[:200_000]
        except Exception:
            return ""

    def _run_static_analysis(self, py_paths: list[str]) -> list[Any]:
        if not py_paths:
            return []
        try:
            from mana_analyzer.analysis.checks import PythonStaticAnalyzer  # type: ignore

            analyzer = PythonStaticAnalyzer()
            all_findings: list[Any] = []
            for rel in py_paths:
                p = (self.repo_root / rel).resolve()
                try:
                    findings = analyzer.analyze_file(p)
                    if findings:
                        all_findings.extend(findings)
                except Exception as exc:
                    all_findings.append({"path": str(rel), "error": f"Static analysis error: {exc}"})
            return all_findings
        except Exception as exc:
            logger.debug("Static analysis unavailable: %s", exc)
            return []