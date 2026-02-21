from __future__ import annotations

from dataclasses import dataclass

from mana_analyzer.analysis.models import Finding
from mana_analyzer.llm.analyze_chain import AnalyzeChain


@dataclass
class _FakeResponse:
    content: str


class _FakeInvoker:
    def __init__(self, content: str) -> None:
        self._content = content

    def invoke(self, _payload: dict) -> _FakeResponse:
        return _FakeResponse(content=self._content)


class _FakePrompt:
    def __init__(self, content: str) -> None:
        self._content = content

    def __or__(self, _other: object) -> _FakeInvoker:
        return _FakeInvoker(content=self._content)


def _chain_with_content(content: str) -> AnalyzeChain:
    chain = AnalyzeChain.__new__(AnalyzeChain)
    chain.prompt = _FakePrompt(content=content)
    chain.llm = object()
    return chain


def test_llm_analyze_chain_parses_and_coerces_findings() -> None:
    chain = _chain_with_content(
        '[{"rule_id":"bug-risk","severity":"CRITICAL","message":"m","line":"-3","column":"-2"}]'
    )
    findings = chain.run("/tmp/a.py", "print(1)\n", [Finding("x", "warning", "m", "/tmp/a.py", 1, 0)])

    assert len(findings) == 1
    assert findings[0].rule_id == "llm-bug-risk"
    assert findings[0].severity == "warning"
    assert findings[0].file_path == "/tmp/a.py"
    assert findings[0].line == 1
    assert findings[0].column == 0


def test_llm_analyze_chain_handles_invalid_json() -> None:
    chain = _chain_with_content("not-json")
    findings = chain.run("/tmp/a.py", "print(1)\n", [])
    assert findings == []


def test_llm_analyze_chain_drops_empty_message_records() -> None:
    chain = _chain_with_content(
        '[{"rule_id":"llm-one","severity":"warning","message":"   ","line":1,"column":0}]'
    )
    findings = chain.run("/tmp/a.py", "print(1)\n", [])
    assert findings == []
