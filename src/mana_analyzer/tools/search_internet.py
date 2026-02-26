"""Internet-search tool backed by Tavily with safe offline fallback."""

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any
from urllib import error, request

logger = logging.getLogger(__name__)

_TAVILY_ENDPOINT = "https://api.tavily.com/search"
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


def safe_search_internet(query: str) -> dict[str, Any]:
    """Search the web with Tavily and return normalized JSON results."""
    query = (query or "").strip()
    if not query:
        return SearchInternetResult(ok=False, query="", results=[], error="Query must not be empty").to_dict()

    api_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if not api_key:
        return SearchInternetResult(
            ok=False,
            query=query,
            results=[],
            error="TAVILY_API_KEY is not configured",
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
    except Exception:  # pragma: no cover
        from langchain.tools import StructuredTool  # type: ignore

    return StructuredTool.from_function(
        func=safe_search_internet,
        name="search_internet",
        description=(
            "Perform a web search via Tavily and return JSON results. "
            "If Tavily is not configured or unreachable, returns ok=false with an error."
        ),
        args_schema=None,
    )
