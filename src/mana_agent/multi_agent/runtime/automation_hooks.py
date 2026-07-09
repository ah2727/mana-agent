"""Automation trigger hooks for multi-agent runtime (Grok Build).

Exposes a minimal, explicit registry for automation triggers.
Hooks are invoked only:
- After explicit model decision (e.g. decision.selected_route or context mentions automation)
- Or from external explicit callers (dashboard trigger buttons, CLI)

No auto-execution, no keyword fallbacks. Complements src.mana_agent.automations.

Update src/mana_agent/multi_agent/ per request to expose hooks.
"""

from __future__ import annotations

from typing import Any, Callable

from mana_agent.multi_agent.core.types import DecisionRecord

__all__ = ["register_automation_trigger", "invoke_automation", "list_triggers"]

# Registry of trigger_name -> callable(context, decision=None) -> result_dict
_AUTOMATION_TRIGGERS: dict[str, Callable[[dict[str, Any], DecisionRecord | None], dict[str, Any] | None]] = {}


def register_automation_trigger(name: str, fn: Callable[[dict[str, Any], DecisionRecord | None], dict[str, Any] | None]) -> None:
    """Register a trigger. Idempotent. Name should be stable (e.g. 'self_improvement')."""
    if not name or not callable(fn):
        return
    _AUTOMATION_TRIGGERS[str(name)] = fn


def list_triggers() -> list[str]:
    return sorted(_AUTOMATION_TRIGGERS.keys())


def invoke_automation(trigger_name: str, context: dict[str, Any], *, decision: DecisionRecord | None = None) -> dict[str, Any] | None:
    """Invoke if registered.

    The caller (work queue post-success, dashboard, etc.) must ensure the invocation
    is backed by a validated model decision or explicit user action.
    """
    fn = _AUTOMATION_TRIGGERS.get(trigger_name)
    if not fn:
        return None
    try:
        return fn(dict(context), decision)
    except Exception as e:
        # Never crash the main flow
        return {"error": str(e), "trigger": trigger_name}


# Default registrations (lazy safe). These delegate to the src automations layer.
def _default_self_improvement(context: dict[str, Any], decision: DecisionRecord | None = None) -> dict[str, Any] | None:
    # Only act if decision present or context signals explicit
    if decision is None and not context.get("explicit_trigger"):
        return {"skipped": "no explicit decision or trigger flag"}
    try:
        from pathlib import Path as _P
        from mana_agent.automations.self_improvement import run_self_improvement_loop  # type: ignore

        r = context.get("root") or "."
        root = r if isinstance(r, _P) else _P(r)
        limit = int(context.get("limit", 3))
        res = run_self_improvement_loop(root, limit=limit) or []
        return {"trigger": "self_improvement", "created_skills": len(res), "items": res}
    except Exception as e:
        return {"error": str(e)}


register_automation_trigger("self_improvement", _default_self_improvement)
register_automation_trigger("improve", _default_self_improvement)