"""Mandatory hierarchical multi-agent execution system."""

from __future__ import annotations

from typing import Any

__all__ = ["MainAgent", "MainAgentResult"]


def __getattr__(name: str) -> Any:
    if name in {"MainAgent", "MainAgentResult"}:
        from mana_agent.multi_agent.agents.main_agent import MainAgent, MainAgentResult

        return {"MainAgent": MainAgent, "MainAgentResult": MainAgentResult}[name]
    raise AttributeError(name)
