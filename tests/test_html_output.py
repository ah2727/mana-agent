from __future__ import annotations

from mana_analyzer.renderers.html_report import (
    render_analyze_html,
    render_describe_html,
    render_report_html,
)


def test_render_analyze_html_contains_sections() -> None:
    html_doc = render_analyze_html(
        {
            "findings": [
                {
                    "severity": "warning",
                    "rule_id": "missing-docstring",
                    "file_path": "src/a.py",
                    "line": 1,
                    "column": 0,
                    "message": "msg",
                }
            ],
            "summarization": {
                "architecture_summary": "arch",
                "tech_summary": "tech",
            },
            "tech": {"languages": ["python"], "file_count": 1},
            "project_structure_analysis": {"line_count": 2, "analysis_lines": ["001. first", "002. second"]},
            "project_root": "/tmp/project",
        },
        "# Analyze",
    )

    assert "<!DOCTYPE html>" in html_doc
    assert "Analyze Report" in html_doc
    assert "missing-docstring" in html_doc
    assert "Project Structure Analysis" in html_doc


def test_render_describe_html_escapes_payload() -> None:
    html_doc = render_describe_html(
        {
            "architecture_summary": "<script>alert(1)</script>",
            "tech_summary": "plain",
            "descriptions": [],
        },
        "# heading",
    )

    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_doc
    assert "<script>alert(1)</script>" not in html_doc


def test_render_report_html_contains_security_and_findings() -> None:
    html_doc = render_report_html(
        {
            "meta": {
                "project_root": "/tmp/project",
                "generated_at": "2025-01-01T00:00:00Z",
                "tool_version": "test",
                "online": True,
                "llm_enabled": True,
                "limitations": ["Only direct dependencies"],
            },
            "summary": {
                "languages": ["python"],
                "frameworks": ["Typer"],
                "technologies": ["Typer"],
                "finding_counts": {"total": 1, "warning": 1, "error": 0},
                "security_counts": {"potential_vulns": 1},
                "status": "review",
            },
            "project_summary": {
                "describe": {"architecture_summary": "arch", "tech_summary": "tech"},
                "file_structure": {"tree_markdown": "src/\n  a.py"},
                "flow_analysis": {"content_markdown": "## Flow\n- step"},
            },
            "findings": {
                "merged_findings": [
                    {
                        "severity": "warning",
                        "rule_id": "missing-docstring",
                        "file_path": "src/a.py",
                        "message": "msg",
                    }
                ],
                "by_rule": {"missing-docstring": 1},
            },
            "security": {
                "vulnerabilities_by_scope": {
                    "runtime": [
                        {
                            "package": {"name": "typer"},
                            "osv_id": "OSV-1",
                            "confidence": "high",
                        }
                    ],
                    "dev": [],
                }
            },
            "warnings": ["Heads up"],
        },
        "# Report",
    )

    assert "Project Audit Report" in html_doc
    assert "missing-docstring" in html_doc
    assert "OSV-1" in html_doc
    assert "Heads up" in html_doc
