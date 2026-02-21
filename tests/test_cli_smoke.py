from pathlib import Path

from typer.testing import CliRunner

from mana_analyzer.analysis.models import AskResponse, AskResponseWithTrace, Finding, SearchHit
from mana_analyzer.commands.cli import app

runner = CliRunner()


class FakeIndexService:
    def index(self, target_path: str, index_dir: Path, rebuild: bool = False) -> dict:
        assert target_path
        assert index_dir
        return {
            "indexed_files": 1,
            "deleted_files": 0,
            "total_files": 1,
            "new_chunks": 2,
            "removed_chunks": 0,
            "index_dir": str(index_dir),
        }


class FakeSearchService:
    def search(self, index_dir: str, query: str, k: int) -> list[SearchHit]:
        assert index_dir
        assert query
        assert k
        return [
            SearchHit(
                score=0.99,
                file_path="/tmp/good.py",
                start_line=1,
                end_line=5,
                symbol_name="add",
                snippet="snippet",
            )
        ]


class FakeAnalyzeService:
    def __init__(self, findings: list[Finding]) -> None:
        self._findings = findings

    def analyze(self, path: str) -> list[Finding]:
        assert path
        return self._findings


class FakeLlmAnalyzeService:
    def __init__(self, findings: list[Finding]) -> None:
        self._findings = findings

    def analyze(self, path: str, static_findings: list[Finding], max_files: int = 10) -> list[Finding]:
        assert path
        assert isinstance(static_findings, list)
        assert max_files > 0
        return self._findings


class FakeAskService:
    def ask(self, index_dir: str, question: str, k: int) -> AskResponse:
        assert index_dir
        assert question
        assert k
        hit = SearchHit(0.8, "/tmp/good.py", 2, 4, "add", "snippet")
        return AskResponse(answer="Uses add. /tmp/good.py:2-4", sources=[hit])

    def ask_with_tools(
        self,
        index_dir: str,
        question: str,
        k: int,
        max_steps: int = 6,
        timeout_seconds: int = 30,
    ) -> AskResponse:
        assert index_dir
        assert question
        assert k
        assert max_steps > 0
        assert timeout_seconds > 0
        hit = SearchHit(0.9, "/tmp/good.py", 1, 3, "add", "snippet")
        return AskResponse(answer="Tool answer. /tmp/good.py:1-3", sources=[hit])

    def ask_dir_mode(self, index_dirs, question: str, k: int, root_dir: str) -> AskResponse:
        assert index_dirs
        assert question
        assert k
        assert root_dir
        hit = SearchHit(0.7, "/tmp/mono/pkg-a/a.py", 1, 2, "a", "snippet")
        return AskResponse(answer="Dir answer", sources=[hit], warnings=[])

    def ask_with_tools_dir_mode(
        self,
        index_dirs,
        question: str,
        k: int,
        max_steps: int = 6,
        timeout_seconds: int = 30,
        root_dir: str | None = None,
    ) -> AskResponseWithTrace:
        assert index_dirs
        assert question
        assert k
        assert max_steps > 0
        assert timeout_seconds > 0
        _ = root_dir
        hit = SearchHit(0.77, "/tmp/mono/pkg-a/a.py", 1, 2, "a", "snippet")
        return AskResponseWithTrace(answer="Dir tool answer", sources=[hit], mode="agent-tools", trace=[], warnings=[])


class DummySettings:
    openai_api_key = "test"
    openai_base_url = None
    openai_chat_model = "fake"
    openai_embed_model = "fake"
    default_top_k = 8


class FakeStructureService:
    def __init__(self, include_tests: bool = False) -> None:
        self.include_tests = include_tests

    def analyze_project(self, target_path: str) -> object:
        assert target_path
        return type(
            "_Report",
            (),
            {
                "to_dict": lambda self: {
                    "project_root": "/tmp/project",
                    "modules": [],
                    "exports": [],
                    "data_structures": [],
                    "commands": [],
                }
            },
        )()

    def render_markdown(self, _report: object) -> str:
        return "# Project Structure Analysis\n\n## Modules\n"


class FakeDependencyReport:
    project_root = "/tmp/project"
    frameworks = ["Typer"]
    technologies = ["Typer", "LangChain"]
    package_managers = ["pip"]
    languages = ["python"]
    runtime_dependencies = ["typer"]
    dev_dependencies = ["pytest"]
    module_edges = []
    dependency_edges = []
    manifests = ["pyproject.toml"]

    def to_dict(self) -> dict:
        return {
            "project_root": self.project_root,
            "frameworks": self.frameworks,
            "technologies": self.technologies,
            "package_managers": self.package_managers,
            "languages": self.languages,
            "runtime_dependencies": self.runtime_dependencies,
            "dev_dependencies": self.dev_dependencies,
            "module_edges": [],
            "dependency_edges": [],
            "manifests": self.manifests,
        }

    def to_dot(self) -> str:
        return 'digraph mana_analyzer { "a" -> "b"; }'

    def to_graphml(self) -> str:
        return "<graphml></graphml>"


class FakeDependencyService:
    def analyze(self, path: str) -> FakeDependencyReport:
        assert path
        return FakeDependencyReport()


class FakeDescribeService:
    def describe(self, path: str, max_files: int = 12, include_functions: bool = False, use_llm: bool = True) -> object:
        assert path
        assert max_files > 0
        _ = include_functions
        _ = use_llm
        return type(
            "_DescribeReport",
            (),
            {
                "to_dict": lambda self: {
                    "project_root": "/tmp/project",
                    "selected_files": ["src/a.py"],
                    "descriptions": [
                        {
                            "file_path": "src/a.py",
                            "language": "python",
                            "symbols": ["add"],
                            "summary": "a summary",
                        }
                    ],
                    "architecture_summary": "arch",
                    "tech_summary": "tech",
                    "chain_steps": ["one", "two"],
                }
            },
        )()

    def render_markdown(self, _report: object) -> str:
        return "# Repository Description\n\n## Architecture\n"


def test_cli_commands(monkeypatch, tmp_path: Path) -> None:
    describe_build_calls: list[bool] = []

    def fake_build_describe_service(_s, model_override=None, use_llm=True):
        _ = model_override
        describe_build_calls.append(use_llm)
        return FakeDescribeService()

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_index_service", lambda _s: FakeIndexService())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_search_service", lambda _s: FakeSearchService())
    monkeypatch.setattr(
        "mana_analyzer.commands.cli.build_analyze_service",
        lambda: FakeAnalyzeService([Finding("missing-docstring", "warning", "msg", "/tmp/a.py", 1, 0)]),
    )
    monkeypatch.setattr(
        "mana_analyzer.commands.cli.build_llm_analyze_service",
        lambda _s, model_override=None: FakeLlmAnalyzeService(
            [Finding("llm-bug-risk", "error", "llm msg", "/tmp/a.py", 2, 0)]
        ),
    )
    monkeypatch.setattr("mana_analyzer.commands.cli.build_ask_service", lambda _s, model_override=None: FakeAskService())
    monkeypatch.setattr("mana_analyzer.commands.cli.StructureService", FakeStructureService)
    monkeypatch.setattr("mana_analyzer.commands.cli.build_dependency_service", lambda: FakeDependencyService())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_describe_service", fake_build_describe_service)
    monkeypatch.setattr("mana_analyzer.commands.cli.discover_subprojects", lambda root: [])
    monkeypatch.setattr("mana_analyzer.commands.cli.discover_index_dirs", lambda root: [Path(root) / ".mana_index"])

    idx = tmp_path / "idx"

    result_index = runner.invoke(app, ["index", str(tmp_path), "--index-dir", str(idx), "--json"])
    assert result_index.exit_code == 0
    assert "indexed_files" in result_index.stdout

    result_search = runner.invoke(app, ["search", "add", "--index-dir", str(idx), "--json"])
    assert result_search.exit_code == 0
    assert "symbol_name" in result_search.stdout

    result_analyze = runner.invoke(app, ["analyze", str(tmp_path), "--fail-on", "error", "--json"])
    assert result_analyze.exit_code == 0

    result_analyze_llm = runner.invoke(
        app,
        [
            "analyze",
            str(tmp_path),
            "--with-llm",
            "--model",
            "gpt-test",
            "--llm-max-files",
            "5",
            "--json",
        ],
    )
    assert result_analyze_llm.exit_code == 0
    assert "llm-bug-risk" in result_analyze_llm.stdout

    result_ask = runner.invoke(app, ["ask", "what", "--index-dir", str(idx), "--json"])
    assert result_ask.exit_code == 0
    assert "answer" in result_ask.stdout

    result_ask_agent = runner.invoke(
        app,
        ["ask", "what", "--index-dir", str(idx), "--agent-tools", "--agent-max-steps", "3", "--json"],
    )
    assert result_ask_agent.exit_code == 0
    assert "Tool answer" in result_ask_agent.stdout

    result_analyze_structure_json = runner.invoke(
        app,
        ["analyze", str(tmp_path), "--full-structure", "--output-format", "json", "--json"],
    )
    assert result_analyze_structure_json.exit_code == 0
    assert "project_root" in result_analyze_structure_json.stdout
    assert "summarization" in result_analyze_structure_json.stdout
    assert "architecture_summary" in result_analyze_structure_json.stdout
    assert "tech_summary" in result_analyze_structure_json.stdout

    result_analyze_structure_md = runner.invoke(
        app,
        ["analyze", str(tmp_path), "--full-structure", "--output-format", "markdown"],
    )
    assert result_analyze_structure_md.exit_code == 0
    assert "Project Structure Analysis" in result_analyze_structure_md.stdout
    assert "Repository Summary" in result_analyze_structure_md.stdout
    assert "Architecture" in result_analyze_structure_md.stdout
    assert "Technology" in result_analyze_structure_md.stdout

    result_analyze_structure_llm = runner.invoke(
        app,
        ["analyze", str(tmp_path), "--with-llm", "--full-structure", "--output-format", "json", "--json"],
    )
    assert result_analyze_structure_llm.exit_code == 0
    assert "summarization" in result_analyze_structure_llm.stdout
    assert describe_build_calls == [False, False, True]

    result_deps = runner.invoke(app, ["deps", str(tmp_path), "--json"])
    assert result_deps.exit_code == 0
    assert "frameworks" in result_deps.stdout

    result_graph = runner.invoke(app, ["graph", str(tmp_path), "--json"])
    assert result_graph.exit_code == 0
    assert "project_root" in result_graph.stdout

    result_describe = runner.invoke(app, ["describe", str(tmp_path), "--no-llm"])
    assert result_describe.exit_code == 0
    assert "Repository Description" in result_describe.stdout

    result_ask_dir_mode = runner.invoke(
        app,
        ["ask", "what", "--dir-mode", "--root-dir", str(tmp_path), "--json"],
    )
    assert result_ask_dir_mode.exit_code == 0
    assert "Dir answer" in result_ask_dir_mode.stdout

    result_ask_dir_mode_tools = runner.invoke(
        app,
        ["ask", "what", "--dir-mode", "--root-dir", str(tmp_path), "--agent-tools", "--json"],
    )
    assert result_ask_dir_mode_tools.exit_code == 0
    assert "Dir tool answer" in result_ask_dir_mode_tools.stdout


def test_analyze_fail_on_warning(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr(
        "mana_analyzer.commands.cli.build_analyze_service",
        lambda: FakeAnalyzeService([Finding("missing-docstring", "warning", "msg", "/tmp/a.py", 1, 0)]),
    )

    result = runner.invoke(app, ["analyze", str(tmp_path), "--fail-on", "warning"])
    assert result.exit_code == 1


def test_analyze_fail_on_merged_findings(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr(
        "mana_analyzer.commands.cli.build_analyze_service",
        lambda: FakeAnalyzeService([]),
    )
    monkeypatch.setattr(
        "mana_analyzer.commands.cli.build_llm_analyze_service",
        lambda _s, model_override=None: FakeLlmAnalyzeService(
            [Finding("llm-generic", "warning", "llm warning", "/tmp/a.py", 1, 0)]
        ),
    )

    result = runner.invoke(app, ["analyze", str(tmp_path), "--with-llm", "--fail-on", "warning"])
    assert result.exit_code == 1


def test_ask_dir_mode_no_auto_index_missing(monkeypatch, tmp_path: Path) -> None:
    class _AskServiceNoIndexes(FakeAskService):
        def ask_dir_mode(self, index_dirs, question: str, k: int, root_dir: str) -> AskResponse:
            assert index_dirs == []
            return AskResponse(answer=f"No usable indexes found under {root_dir}", sources=[], warnings=[])

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_ask_service", lambda _s, model_override=None: _AskServiceNoIndexes())
    monkeypatch.setattr("mana_analyzer.commands.cli.discover_index_dirs", lambda root: [])

    class _Sub:
        def __init__(self, root_path: Path) -> None:
            self.root_path = root_path

    monkeypatch.setattr("mana_analyzer.commands.cli.discover_subprojects", lambda root: [_Sub(Path(root) / "pkg-a")])

    result = runner.invoke(
        app,
        ["ask", "what", "--dir-mode", "--root-dir", str(tmp_path), "--no-auto-index-missing", "--json"],
    )
    assert result.exit_code == 0
    assert "No usable indexes found under" in result.stdout
