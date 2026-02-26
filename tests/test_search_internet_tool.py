from __future__ import annotations

import json
from urllib.error import HTTPError

from mana_analyzer.tools.search_internet import safe_search_internet


def test_safe_search_internet_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    payload = safe_search_internet("latest release")
    assert payload["ok"] is False
    assert payload["results"] == []
    assert "TAVILY_API_KEY" in payload["error"]


def test_safe_search_internet_success(monkeypatch) -> None:
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "results": [
                        {
                            "title": "Mana Analyzer",
                            "url": "https://example.com/mana",
                            "content": "release notes",
                            "score": 0.88,
                        }
                    ]
                }
            ).encode("utf-8")

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr("mana_analyzer.tools.search_internet.request.urlopen", lambda *a, **k: _Response())

    payload = safe_search_internet("mana analyzer latest")
    assert payload["ok"] is True
    assert payload["query"] == "mana analyzer latest"
    assert len(payload["results"]) == 1
    assert payload["results"][0]["title"] == "Mana Analyzer"
    assert payload["results"][0]["url"] == "https://example.com/mana"


def test_safe_search_internet_http_error(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    def _raise_http_error(*_args, **_kwargs):
        raise HTTPError(url="https://api.tavily.com/search", code=500, msg="bad", hdrs=None, fp=None)

    monkeypatch.setattr("mana_analyzer.tools.search_internet.request.urlopen", _raise_http_error)

    payload = safe_search_internet("latest")
    assert payload["ok"] is False
    assert payload["results"] == []
    assert "HTTP error" in payload["error"]
