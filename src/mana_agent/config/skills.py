"""Typed adaptive-skill configuration with environment-safe defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from mana_agent.skills.adaptive import skills_root


def _flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class AdaptiveSkillsConfig:
    enabled: bool = True
    root_path: Path | None = None
    progressive_loading: bool = True
    max_loaded_per_task: int = 4
    activation: str = "ask"
    generation: bool = True
    record_usage: bool = True
    chat_enabled: bool = True
    show_selection_events: bool = True

    @classmethod
    def from_environment(cls) -> "AdaptiveSkillsConfig":
        maximum = int(os.getenv("MANA_SKILLS_MAX_LOADED", "4"))
        if maximum < 1: raise ValueError("MANA_SKILLS_MAX_LOADED must be at least 1")
        activation = os.getenv("MANA_SKILLS_ACTIVATION", "ask").strip().lower()
        if activation not in {"disabled", "manual", "ask", "auto_verified"}: raise ValueError("invalid MANA_SKILLS_ACTIVATION")
        return cls(
            enabled=_flag("MANA_SKILLS_ENABLED", True), root_path=skills_root(),
            max_loaded_per_task=maximum, activation=activation,
            chat_enabled=_flag("MANA_CHAT_SKILLS_ENABLED", True),
            show_selection_events=_flag("MANA_CHAT_SKILLS_SHOW_EVENTS", True),
        )
