from __future__ import annotations

from mana_agent.multi_agent.agents.base_agent import BaseAgent


class SummarizerAgent(BaseAgent):
    def summarize(self, task_id: str) -> str:
        task = self.taskboard.get_task(task_id)
        files = ", ".join(task.files_touched) if task.files_touched else "none recorded"
        verification = task.verification_results[-1].summary if task.verification_results else "not run yet"
        summary = (
            f"Multi-agent route `{task.status.value}` recorded for task {task.task_id}. "
            f"Files touched: {files}. Verification: {verification}."
        )
        self.record_evidence(task_id, "summary.created")
        return summary
