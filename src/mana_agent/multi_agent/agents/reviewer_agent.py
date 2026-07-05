from mana_agent.multi_agent.agents.base_agent import BaseAgent


class ReviewerAgent(BaseAgent):
    def review(self, task_id: str, risk_summary: str) -> None:
        self.record_evidence(task_id, f"Reviewer assessment: {risk_summary}")
