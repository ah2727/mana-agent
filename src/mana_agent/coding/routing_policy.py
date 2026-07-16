"""Validation helpers for model-produced backend decisions."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from mana_agent.coding.models import CodingBackendDecision
from mana_agent.coding.registry import CodingBackendDecisionError


def validate_backend_decision(payload: dict[str, Any]) -> CodingBackendDecision:
    try:
        return CodingBackendDecision.model_validate(payload)
    except ValidationError as exc:
        raise CodingBackendDecisionError(
            f"Model decision failed: coding_backend. No backend was executed. Reason: {exc}"
        ) from exc


__all__ = ["validate_backend_decision"]
