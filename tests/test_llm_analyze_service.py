from __future__ import annotations

from pathlib import Path

from mana_analyzer.analysis.models import Finding
from mana_analyzer.services.llm_analyze_service import LlmAnalyzeService


class FakeAnalyzeChain:
    def __init__(self, findings_by_file: dict[str, list[Finding]]) -> None:
        self.findings_by_file = findings_by_file
        self.calls: list[str] = []

    def run(self, file_path: str, source: str, static_findings: list[Finding]) -> list[Finding]:
        assert source
        assert isinstance(static_findings, list)
        self.calls.append(file_path)
        return self.findings_by_file.get(file_path, [])


def test_llm_service_selects_top_files_by_static_findings(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    c = tmp_path / "c.py"
    for file in (a, b, c):
        file.write_text("x=1\n", encoding="utf-8")

    static_findings = [
        Finding("missing-docstring", "warning", "x", str(a), 1, 0),
        Finding("missing-docstring", "warning", "x", str(a), 2, 0),
        Finding("missing-docstring", "warning", "x", str(b), 1, 0),
    ]
    chain = FakeAnalyzeChain(
        {
            str(a): [Finding("llm-generic", "warning", "a", str(a), 1, 0)],
            str(b): [Finding("llm-generic", "warning", "b", str(b), 1, 0)],
        }
    )
    service = LlmAnalyzeService(chain)

    findings = service.analyze(tmp_path, static_findings=static_findings, max_files=2)

    assert [Path(path).name for path in chain.calls] == ["a.py", "b.py"]
    assert len(findings) == 2


def test_llm_service_fallback_when_no_static_findings(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("x=1\n", encoding="utf-8")
    b.write_text("x=2\n", encoding="utf-8")

    chain = FakeAnalyzeChain(
        {
            str(a): [Finding("llm-generic", "warning", "a", str(a), 1, 0)],
            str(b): [Finding("llm-generic", "warning", "b", str(b), 1, 0)],
        }
    )
    service = LlmAnalyzeService(chain)

    findings = service.analyze(tmp_path, static_findings=[], max_files=1)

    assert [Path(path).name for path in chain.calls] == ["a.py"]
    assert len(findings) == 1


def test_llm_service_dedupes_findings(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    a.write_text("x=1\n", encoding="utf-8")

    duplicate = Finding("llm-generic", "warning", "dup", str(a), 1, 0)
    chain = FakeAnalyzeChain({str(a): [duplicate, duplicate]})
    service = LlmAnalyzeService(chain)

    findings = service.analyze(tmp_path, static_findings=[], max_files=10)
    assert len(findings) == 1
