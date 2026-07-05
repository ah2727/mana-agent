from __future__ import annotations

import subprocess
from pathlib import Path

from mana_agent.multi_agent.core.ids import new_message_id
from mana_agent.multi_agent.core.types import QueueJob, QueueJobType, ToolResult
from mana_agent.multi_agent.tools.permissions import assert_shell_allowed


class ToolsManager:
    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()

    def execute_job(self, job: QueueJob) -> ToolResult:
        try:
            if job.job_type == QueueJobType.GIT_STATUS:
                return self._shell(job, "git status --short")
            if job.job_type == QueueJobType.GIT_DIFF:
                return self._shell(job, "git diff")
            if job.job_type in {QueueJobType.SHELL, QueueJobType.RUN_TESTS, QueueJobType.RUN_LINT}:
                return self._shell(job, str(job.payload.get("command", "")))
            if job.job_type == QueueJobType.REPO_READ:
                path = self._resolve_path(str(job.payload.get("path", "")))
                return ToolResult(new_message_id(), job.task_id, True, {"content": path.read_text(encoding="utf-8"), "path": str(path)})
            if job.job_type == QueueJobType.REPO_SEARCH:
                query = str(job.payload.get("query", ""))
                result = subprocess.run(["rg", "-n", query, str(self.root)], cwd=self.root, text=True, capture_output=True, timeout=30)
                return ToolResult(new_message_id(), job.task_id, result.returncode in {0, 1}, {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode})
            return ToolResult(new_message_id(), job.task_id, False, error=f"unsupported tool job: {job.job_type.value}")
        except Exception as exc:
            return ToolResult(new_message_id(), job.task_id, False, error=str(exc))

    def _shell(self, job: QueueJob, command: str) -> ToolResult:
        assert_shell_allowed(command)
        result = subprocess.run(command, cwd=self.root, text=True, capture_output=True, shell=True, timeout=120)
        return ToolResult(new_message_id(), job.task_id, result.returncode == 0, {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}, None if result.returncode == 0 else result.stderr)

    def _resolve_path(self, path: str) -> Path:
        resolved = (self.root / path).resolve()
        if self.root not in resolved.parents and resolved != self.root:
            raise ValueError("path escapes repository root")
        return resolved
