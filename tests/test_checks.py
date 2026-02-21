from pathlib import Path

from mana_analyzer.analysis.checks import PythonStaticAnalyzer


def test_static_checks_find_core_issues() -> None:
    fixture = Path(__file__).parent / "fixtures" / "sample_project" / "bad_module.py"
    analyzer = PythonStaticAnalyzer()

    findings = analyzer.analyze_file(fixture)
    rules = [item.rule_id for item in findings]

    assert "wildcard-import" in rules
    assert "unused-imports" in rules
    assert "deep-nesting" in rules
    assert "missing-docstring" in rules


def test_missing_docstring_rule() -> None:
    fixture = Path(__file__).parent / "fixtures" / "sample_project" / "no_doc.py"
    analyzer = PythonStaticAnalyzer()

    findings = analyzer.analyze_file(fixture)
    missing = [item for item in findings if item.rule_id == "missing-docstring"]
    assert missing
