from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MenuOption:
    value: str
    label: str
    aliases: tuple[str, ...] = ()


def select_option(
    *,
    title: str,
    text: str,
    options: Iterable[MenuOption],
    input_func: Callable[[str], str] | None = None,
) -> str:
    """Render a simple terminal-native menu and return the selected value."""
    items = list(options)
    if input_func is None:
        input_func = input
    lines = [title, text]
    for index, option in enumerate(items, start=1):
        lines.append(f"{index}. {option.label}")
    raw = str(input_func("\n".join(lines) + "\n> ") or "").strip()
    lowered = raw.lower()
    for index, option in enumerate(items, start=1):
        choices = {str(index), option.value.lower(), *(alias.lower() for alias in option.aliases)}
        if lowered in choices:
            return option.value
    return raw
