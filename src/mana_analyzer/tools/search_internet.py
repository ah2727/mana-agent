"""Internet-search tool backed by Tavily with safe offline fallback and DuckDuckGo support."""

import json
import logging
import os
import urllib.parse
from dataclasses import asdict, dataclass
from typing import Any
from urllib import error, request

logger = logging.getLogger(__name__)

_TAVILY_ENDPOINT = "https://api.tavily.com/search"
_DUCKDUCKGO_ENDPOINT = "https://api.duckduckgo.com/"
_REQUEST_TIMEOUT_SECONDS = 12
_MAX_RESULTS = 5


@dataclass(frozen=True)
class SearchInternetResult:
    """Result container for the internet-search tool.

    ``ok`` indicates whether a real search was performed.
    ``results`` contains normalized search hit objects.
    ``error`` contains a user-readable failure reason.
    """

    ok: bool
    query: str
    results: list[dict[str, Any]] | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary payload."""
        data = asdict(self)
        if data["results"] is None:
            data["results"] = []
        return data


def _duckduckgo_search(query: str) -> list[dict[str, Any]]:
    """Fallback search using DuckDuckGo's instant answer API.

    Returns a list of normalized result dicts similar to the Tavily format.
    """
    # Build request URL
    params = {
        "q": query,
        "format": "json",
        "no_redirect": "1",
        "no_html": "1",
        "skip_disambig": "1",
    }
    url = f"{_DUCKDUCKGO_ENDPOINT}?{urllib.parse.urlencode(params)}"
    req = request.Request(url, method="GET", headers={"User-Agent": "mana-analyzer/1.0"})
    try:
        with request.urlopen(req, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
    except Exception as exc:  # pragma: no cover
        logger.warning("DuckDuckGo search failed: %s", exc)
        return []

    # DuckDuckGo returns results in 'RelatedTopics' list; each may have 'Text' and 'FirstURL'
    results = []
    for item in data.get("RelatedTopics", [])[:_MAX_RESULTS]:
        # Some items are nested dicts with a 'Topics' list
        if "Topics" in item:
            for sub in item.get("Topics", [])[: _MAX_RESULTS - len(results)]:
                results.append({
                    "title": sub.get("Text", "").split(" - ")[0],
                    "url": sub.get("FirstURL", ""),
                    "content": sub.get("Text", ""),
                    "score": 0.0,
                    "raw": sub,
                })
        else:
            results.append({
                "title": item.get("Text", "").split(" - ")[0],
                "url": item.get("FirstURL", ""),
                "content": item.get("Text", ""),
                "score": 0.0,
                "raw": item,
            })
        if len(results) >= _MAX_RESULTS:
            break
    return results


def safe_search_internet(query: str) -> dict[str, Any]:
    """Search the web with Tavily (or DuckDuckGo fallback) and return normalized JSON results."""
    query = (query or "").strip()
    if not query:
        return SearchInternetResult(ok=False, query="", results=[], error="Query must not be empty").to_dict()

    api_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if not api_key:
        # Fallback to DuckDuckGo public API
        fallback_results = _duckduckgo_search(query)
        if fallback_results:
            return SearchInternetResult(
                ok=True,
                query=query,
                results=fallback_results,
                error="",
            ).to_dict()
        else:
            return SearchInternetResult(
                ok=False,
                query=query,
                results=[],
                error="DuckDuckGo search failed or returned no results",
            ).to_dict()

    try:
        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": _MAX_RESULTS,
            "search_depth": "basic",
            "include_answer": False,
            "include_images": False,
            "include_raw_content": False,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            _TAVILY_ENDPOINT,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
            raw = resp.read().decode("utf-8")
        decoded = json.loads(raw)
        rows = decoded.get("results") or []
        normalized: list[dict[str, Any]] = []
        for row in rows:
            normalized.append(
                {
                    "title": str(row.get("title", "")),
                    "url": str(row.get("url", "")),
                    "content": str(row.get("content", "")),
                    "score": float(row.get("score") or 0.0),
                    "raw": row,
                }
            )

        logger.debug("Internet search returned %d results for query=%r", len(normalized), query)
        return SearchInternetResult(
            ok=True,
            query=query,
            results=normalized,
            error="",
        ).to_dict()
    except error.HTTPError as exc:
        message = f"Tavily HTTP error {exc.code}"
        logger.warning("safe_search_internet http error: %s", message)
        return SearchInternetResult(ok=False, query=query, results=[], error=message).to_dict()
    except error.URLError as exc:
        message = f"Tavily network error: {exc.reason}"
        logger.warning("safe_search_internet url error: %s", message)
        return SearchInternetResult(ok=False, query=query, results=[], error=message).to_dict()
    except Exception as exc:  # pragma: no cover – defensive
        logger.exception("safe_search_internet failed")
        return SearchInternetResult(ok=False, query=query, results=[], error=str(exc)).to_dict()


def build_search_internet_tool() -> "StructuredTool":  # type: ignore[name-defined]
    """Wrap ``safe_search_internet`` in a LangChain ``StructuredTool``."""
    try:
        from langchain_core.tools import StructuredTool  # type: ignore
    except Exception:
        from langchain.tools import StructuredTool  # type: ignore

    return StructuredTool.from_function(
        func=safe_search_internet,
        name="search_internet",
        description=(
            "Perform a web search via Tavily; if Tavily is not configured or unreachable, fallback to DuckDuckGo and return JSON results. "
            "If both fail, returns ok=false with an error."
        ),
        args_schema=None,
    )
