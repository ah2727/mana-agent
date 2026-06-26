"""Supported output formats for the chat ``/analyze`` slash command.

This module is the single source of truth for the analyze artifact formats,
their aliases, the artifact filenames written under ``.mana/``, and the parsing
of both the direct command line (``/analyze json markdown``) and the numbered
menu (``1,2,3``). Keeping this here means the chat handler, the renderer, and
the tests all agree on the same mapping.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = [
    "ANALYZE_ARTIFACTS",
    "ANALYZE_ALIASES",
    "MENU_FORMATS",
    "MENU_NUMBER_MAP",
    "SUPPORTED_TOKENS",
    "UnknownAnalyzeFormat",
    "canonical_formats",
    "parse_analyze_formats",
    "parse_menu_choice",
    "supported_formats_line",
]


# Canonical format key -> artifact filename written under the project ``.mana/``.
# The insertion order here defines the order used when the user requests ``all``.
ANALYZE_ARTIFACTS: dict[str, str] = {
    "json": "analyze.json",
    "markdown": "analyze.md",
    "html": "analyze.html",
    "dot": "analyze.dot",
    "graphml": "analyze.graphml",
    "mermaid": "diagram.mmd",
}

# User-typed token -> canonical key. ``md`` is an alias of ``markdown``.
ANALYZE_ALIASES: dict[str, str] = {
    "json": "json",
    "markdown": "markdown",
    "md": "markdown",
    "html": "html",
    "dot": "dot",
    "graphml": "graphml",
    "mermaid": "mermaid",
}

# Tokens advertised to the user in error/help text (includes ``all``).
SUPPORTED_TOKENS: list[str] = [
    "json",
    "markdown",
    "md",
    "html",
    "dot",
    "graphml",
    "mermaid",
    "all",
]

# Order shown in the interactive menu.
MENU_FORMATS: list[str] = ["json", "markdown", "html", "dot", "graphml", "mermaid"]

# Menu number -> canonical key (7 means "all").
MENU_NUMBER_MAP: dict[int, str] = {
    1: "json",
    2: "markdown",
    3: "html",
    4: "dot",
    5: "graphml",
    6: "mermaid",
    7: "all",
}


class UnknownAnalyzeFormat(ValueError):
    """Raised when the user requests a format that is not supported."""

    def __init__(self, token: str) -> None:
        self.token = token
        super().__init__(f"Unknown analyze format: {token}")


def supported_formats_line() -> str:
    """Return the comma-joined list of supported tokens for error messages."""
    return ", ".join(SUPPORTED_TOKENS)


def _all_formats() -> list[str]:
    return list(ANALYZE_ARTIFACTS.keys())


def _dedupe(formats: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for fmt in formats:
        if fmt not in seen:
            seen.add(fmt)
            ordered.append(fmt)
    return ordered


def canonical_formats(tokens: Iterable[str]) -> list[str]:
    """Map raw tokens to canonical format keys.

    ``all`` expands to every supported format. Raises ``UnknownAnalyzeFormat``
    on the first token that is not recognised.
    """
    resolved: list[str] = []
    for raw in tokens:
        token = str(raw or "").strip().lower()
        if not token:
            continue
        if token == "all":
            resolved.extend(_all_formats())
            continue
        canonical = ANALYZE_ALIASES.get(token)
        if canonical is None:
            raise UnknownAnalyzeFormat(token)
        resolved.append(canonical)
    return _dedupe(resolved)


def parse_analyze_formats(args: str | Iterable[str] | None) -> list[str]:
    """Parse the argument portion of ``/analyze`` into canonical formats.

    Accepts the text after ``/analyze`` (e.g. ``"json markdown"``,
    ``"--format json,markdown,html"``, ``"all"``) or an already-split token
    list. Returns an empty list when no formats are supplied (the caller then
    opens the interactive menu). Raises ``UnknownAnalyzeFormat`` for bad tokens.
    """
    if args is None:
        return []
    if isinstance(args, str):
        text = args
    else:
        text = " ".join(str(item) for item in args)

    # Normalise --format / -f flags and comma separators into plain spaces.
    text = text.strip()
    if not text:
        return []
    text = text.replace("--format", " ").replace("-f", " ")
    text = text.replace("=", " ").replace(",", " ")
    tokens = [tok for tok in text.split() if tok]
    return canonical_formats(tokens)


def parse_menu_choice(raw: str) -> list[str]:
    """Parse a numbered menu response (e.g. ``"1"``, ``"1,2,3"``, ``"7"``).

    Returns the canonical formats for the chosen numbers. Empty input returns
    an empty list (treated as "cancelled" by the caller). Raises ``ValueError``
    with a helpful message for any out-of-range or non-numeric entry.
    """
    text = str(raw or "").strip()
    if not text:
        return []
    text = text.replace(",", " ")
    chosen: list[str] = []
    for piece in text.split():
        try:
            number = int(piece)
        except ValueError as exc:
            raise ValueError(f"Invalid choice: {piece}") from exc
        if number not in MENU_NUMBER_MAP:
            raise ValueError(f"Choice out of range: {number}")
        canonical = MENU_NUMBER_MAP[number]
        if canonical == "all":
            chosen.extend(_all_formats())
        else:
            chosen.append(canonical)
    return _dedupe(chosen)
