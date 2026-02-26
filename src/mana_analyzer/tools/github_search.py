"""GitHub code‑search tool.

This module provides a thin wrapper around the public GitHub *code search* API. It
exposes a ``safe_github_search`` function that returns a JSON‑serialisable payload
compatible with the other tool implementations in the project. The function
behaves gracefully when a ``GITHUB_TOKEN`` environment variable is not provided
or when the request fails – it returns ``ok=False`` together with a human readable
``error`` description.

The tool is registered via :func:`build_github_search_tool` which creates a
``langchain`` :class:`~langchain_core.tools.StructuredTool` (or the legacy
``langchain.tools.StructuredTool`` fallback) wrapping ``safe_github_search``.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
from dataclasses import asdict, dataclass
from typing import Any, Final
from urllib import error, request

logger = logging.getLogger(__name__)

# GitHub API constants -------------------------------------------------------
_GITHUB_SEARCH_ENDPOINT: Final = "https://api.github.com/search/code"
_REQUEST_TIMEOUT_SECONDS: Final = 12
_MAX_RESULTS: Final = 5


@dataclass(frozen=True)
class GitHubSearchResult:
    """Container for the ``safe_github_search`` payload.

    The shape mirrors :class:`mana_analyzer.tools.search_internet.SearchInternetResult`
    so callers can treat both tools uniformly.
    """

    ok: bool
    query: str
    results: list[dict[str, Any]] | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover – trivial
        data = asdict(self)
        if data["results"] is None:
            data["results"] = []
        return data


def _github_code_search(query: str) -> list[dict[str, Any]]:
    """Perform a raw GitHub code search.

    The function extracts a few useful fields from the API response and normalises
    them to the same schema used by the internet‑search tool.
    """
    params = {"q": query, "per_page": str(_MAX_RESULTS)}
    url = f"{_GITHUB_SEARCH_ENDPOINT}?{urllib.parse.urlencode(params)}"

    headers: dict[str, str] = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "mana-analyzer/1.0",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token.strip()}"

    req = request.Request(url, method="GET", headers=headers)
    try:
        with request.urlopen(req, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
            raw = resp.read().decode("utf-8")
        payload = json.loads(raw)
    except Exception as exc:  # pragma: no cover – exercised via tests
        logger.warning("GitHub search failed: %s", exc)
        raise

    items = payload.get("items", [])
    results: list[dict[str, Any]] = []
    for item in items:
        repo_full = item.get("repository", {}).get("full_name", "")
        path = item.get("path", "")
        title = f"{repo_full}/{path}" if repo_full else path
        results.append(
            {
                "title": title,
                "url": item.get("html_url", ""),
                "content": path,
                "score": float(item.get("score", 0.0)),
                "raw": item,
            }
        )
        if len(results) >= _MAX_RESULTS:
            break
    return results


def safe_github_search(query: str) -> dict[str, Any]:
    """Public entry point wrapped as a LangChain ``StructuredTool``.

    Mirrors the behaviour of ``safe_search_internet``: validates the query,
    performs the request, normalises the output and guarantees a dictionary with
    the keys ``ok``, ``query``, ``results`` and ``error``.
    """
    query = (query or "").strip()
    if not query:
        return GitHubSearchResult(ok=False, query="", results=[], error="Query must not be empty").to_dict()

    try:
        raw_results = _github_code_search(query)
        return GitHubSearchResult(ok=True, query=query, results=raw_results, error="").to_dict()
    except Exception as exc:  # pragma: no cover – exercised via integration tests
        return GitHubSearchResult(ok=False, query=query, results=[], error=str(exc)).to_dict()


def build_github_search_tool() -> "StructuredTool":  # type: ignore[name-defined]
    """Wrap :func:`safe_github_search` in a LangChain ``StructuredTool``.

    Follows the pattern used for the internet‑search tool so that the rest of the
    codebase can treat both uniformly.
    """
    try:
        from langchain_core.tools import StructuredTool  # type: ignore
    except Exception:  # pragma: no cover – fallback for older LangChain versions
        from langchain.tools import StructuredTool  # type: ignore

    return StructuredTool.from_function(
        func=safe_github_search,
        name="github_code_search",
        description=(
            "Search public GitHub code using the GitHub Search API. Returns a JSON "
            "payload with ``ok``, ``query``, a list of ``results`` (each containing "
            "title, url, content, score) and an ``error`` field when the request fails."
        ),
        args_schema=None,
    )
