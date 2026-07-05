from mana_agent.multi_agent.agents.base_agent import BaseAgent


class ResearchAgent(BaseAgent):
    def collect_evidence(self, task_id: str, summary: str) -> None:
        self.record_evidence(task_id, f"Research evidence: {summary}")
