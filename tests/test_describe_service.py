from __future__ import annotations

from pathlib import Path

from mana_analyzer.services.dependency_service import DependencyService
from mana_analyzer.services.describe_service import DescribeService


class _FailingArchitectureChain:
    def summarize_file(self, file_path: Path, language: str, source: str) -> tuple[str, list[str]]:
        return (f"Summary for {file_path.name}", ["Service.run"])

    def synthesize_architecture(self, dependency_report: dict, file_summaries: list[dict]) -> tuple[str, str]:
        raise RuntimeError("provider rejected request")


def _write_repo(root: Path) -> None:
    (root / "app").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        """
[project]
dependencies = ["fastapi>=0.100"]
""".strip(),
        encoding="utf-8",
    )
    (root / "app" / "main.py").write_text(
        """
class Service:
    def run(self) -> int:
        return 1

def handler() -> int:
    return Service().run()
""".strip(),
        encoding="utf-8",
    )


def test_describe_service_without_llm(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    service = DescribeService(dependency_service=DependencyService(), llm_chain=None)

    report = service.describe(tmp_path, max_files=5, include_functions=True, use_llm=False)

    assert report.project_root == str(tmp_path)
    assert report.descriptions
    assert "dependency-analysis" in report.chain_steps
    assert "Technologies:" in report.tech_summary


def test_describe_markdown(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    service = DescribeService(dependency_service=DependencyService(), llm_chain=None)
    report = service.describe(tmp_path, max_files=5, include_functions=False, use_llm=False)

    markdown = service.render_markdown(report)

    assert "# Repository Description" in markdown
    assert "## Architecture" in markdown


def test_describe_service_falls_back_when_llm_architecture_fails(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    service = DescribeService(dependency_service=DependencyService(), llm_chain=_FailingArchitectureChain())

    report = service.describe(tmp_path, max_files=5, include_functions=True, use_llm=True)

    assert report.descriptions
    assert report.architecture_summary.startswith("Repository contains")
    assert report.tech_summary.startswith("Technologies:")
