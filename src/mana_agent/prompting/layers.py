from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptLayer:
    name: str
    content: str


PROMPT_LAYER_ORDER: tuple[str, ...] = (
    "core_identity",
    "tool_rules",
    "mode_rules",
    "skills_index",
    "memory_snapshot",
    "task_context",
    "output_contract",
)


def compose_layers(layers: list[PromptLayer]) -> str:
    names = tuple(layer.name for layer in layers)
    if names != PROMPT_LAYER_ORDER:
        expected = " -> ".join(PROMPT_LAYER_ORDER)
        actual = " -> ".join(names)
        raise ValueError(f"prompt layers must follow stable order: {expected}; got: {actual}")
    return "\n\n".join(layer.content.strip() for layer in layers if layer.content.strip()).strip()
