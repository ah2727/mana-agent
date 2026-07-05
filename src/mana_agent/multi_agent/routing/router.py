from __future__ import annotations

from mana_agent.multi_agent.core.types import RiskLevel, RouteDecision
from mana_agent.multi_agent.routing.policies import classify_request


class Router:
    def route(self, *, task_id: str, user_request: str) -> RouteDecision:
        kind = classify_request(user_request)
        if kind == "coding":
            return RouteDecision(task_id, "coding", ["main", "head_decision", "planner", "coding", "tool", "verifier", "reviewer", "summarizer"], ["planning", "coding", "tool_execution", "verification", "review", "summarization"], True, True, RiskLevel.MEDIUM, "Code mutation or repository edit request.")
        if kind == "analyze":
            return RouteDecision(task_id, "analyze", ["main", "head_decision", "research", "planner", "reviewer", "summarizer"], ["repo_search", "repo_read", "planning", "review", "summarization"], True, False, RiskLevel.LOW, "Repository analysis request.")
        if kind == "planning":
            return RouteDecision(task_id, "planning", ["main", "head_decision", "planner", "reviewer", "summarizer"], ["planning", "review", "summarization"], True, False, RiskLevel.LOW, "Planning request.")
        if kind == "high_risk_tool":
            return RouteDecision(task_id, "high_risk_tool", ["main", "head_decision", "tool"], ["decision", "tool_execution"], True, True, RiskLevel.HIGH, "High-risk shell or git operation requires approval.")
        if kind == "tool":
            return RouteDecision(task_id, "tool", ["main", "head_decision", "tool", "verifier", "summarizer"], ["tool_execution", "verification", "summarization"], False, True, RiskLevel.MEDIUM, "Tool-heavy request.")
        return RouteDecision(task_id, "simple", ["main", "summarizer"], ["conversation", "summarization"], False, False, RiskLevel.LOW, "Simple explanation or Q&A request.")
