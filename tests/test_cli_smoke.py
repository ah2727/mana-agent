import json
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
    coding_flow_max_turns = 5
    coding_flow_max_tasks = 20
    coding_plan_max_steps = 8
    coding_search_budget = 4
    coding_read_budget = 6
    coding_require_read_files = 2


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
    def describe(
        self,
        path: str,
        max_files: int = 12,
        include_functions: bool = False,
        use_llm: bool = True,
        **_: object,
    ) -> object:
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
                    "architecture_mermaid": "flowchart LR",
                    "architecture_data": {},
                    "metrics": {},
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


def test_build_ask_service_registers_search_internet_tool_without_duplicates(monkeypatch, tmp_path: Path) -> None:
    from mana_analyzer.commands import cli

    class _Tool:
        def __init__(self, name: str) -> None:
            self.name = name

    class _FakeAskAgent:
        def __init__(self, **_: object) -> None:
            # Simulate pre-existing registration.
            self.tools = [_Tool("search_internet")]

    monkeypatch.setattr("mana_analyzer.commands.cli.AskAgent", _FakeAskAgent)
    monkeypatch.setattr("mana_analyzer.commands.cli.QnAChain", lambda **_: object())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_store", lambda _s: object())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_search_service", lambda _s: object())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_search_internet_tool", lambda: _Tool("search_internet"))

    svc = cli.build_ask_service(DummySettings(), model_override=None, project_root=tmp_path)
    assert svc.ask_agent is not None
    assert sum(1 for tool in svc.ask_agent.tools if getattr(tool, "name", "") == "search_internet") == 1


def test_chat_blocks_edit_requests_without_coding_agent(monkeypatch, tmp_path: Path) -> None:
    class _NoCallAskService(FakeAskService):
        def ask(self, index_dir: str, question: str, k: int) -> AskResponse:  # pragma: no cover - must not run
            raise AssertionError("chat_service.ask should not be called for blocked edit requests")

        def ask_with_tools(  # pragma: no cover - must not run
            self,
            index_dir: str,
            question: str,
            k: int,
            max_steps: int = 6,
            timeout_seconds: int = 30,
        ) -> AskResponse:
            raise AssertionError("ask_with_tools should not be called for blocked edit requests")

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_ask_service", lambda _s, model_override=None: _NoCallAskService())

    result = runner.invoke(
        app,
        ["chat", "--agent-tools"],
        input="please patch this file\nquit\n",
    )
    assert result.exit_code == 0
    assert "read-only for file edits" in result.stdout
    assert "--agent-tools" in result.stdout
    assert "--coding-agent" in result.stdout


def test_chat_transparency_sections_always_render_in_normal_mode(monkeypatch, tmp_path: Path) -> None:
    class _AskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_ask_service", lambda _s, model_override=None: _AskService())

    result = runner.invoke(
        app,
        ["chat"],
        input="first question\nsecond question\nquit\n",
    )
    assert result.exit_code == 0
    assert result.stdout.count("Summary") >= 2
    assert result.stdout.count("Steps") >= 2
    assert result.stdout.count("Decisions") >= 2
    assert result.stdout.count("History") >= 2
    assert "Session History" in result.stdout


def test_chat_transparency_uses_trace_steps_in_agent_tools_mode(monkeypatch, tmp_path: Path) -> None:
    class _TracingAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

        def ask_with_tools(
            self,
            index_dir: str,
            question: str,
            k: int,
            max_steps: int = 6,
            timeout_seconds: int = 30,
        ) -> AskResponseWithTrace:
            _ = (index_dir, question, k, max_steps, timeout_seconds)
            return AskResponseWithTrace(
                answer="Decision: Use semantic search first",
                sources=[],
                mode="agent-tools",
                trace=[
                    {
                        "tool_name": "semantic_search",
                        "status": "ok",
                        "duration_ms": 3.5,
                        "args_summary": "query='planner'",
                    }
                ],
                warnings=[],
            )

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_ask_service", lambda _s, model_override=None: _TracingAskService())

    result = runner.invoke(
        app,
        ["chat", "--agent-tools"],
        input="plan this\nquit\n",
    )
    assert result.exit_code == 0
    assert "Steps" in result.stdout
    assert "semantic_search" in result.stdout
    assert "Decisions" in result.stdout
    assert "Use semantic search first" in result.stdout


def test_chat_writes_llm_run_log_rows(monkeypatch, tmp_path: Path) -> None:
    class _AskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    rows: list[dict] = []

    class _FakeRunLogger:
        def __init__(self, log_file=None) -> None:
            _ = log_file

        def log(self, payload: dict) -> None:
            rows.append(payload)

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_ask_service", lambda _s, model_override=None: _AskService())
    monkeypatch.setattr("mana_analyzer.commands.cli.LlmRunLogger", _FakeRunLogger)

    result = runner.invoke(
        app,
        ["chat"],
        input="what is this project?\nquit\n",
    )
    assert result.exit_code == 0
    assert rows
    assert rows[0]["flow"] == "chat"
    assert rows[0]["mode"] == "classic"
    assert rows[0]["question"] == "what is this project?"


def test_flow_show_checkpoint_and_reset_commands(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-123"
            self.checkpointed: list[str] = []
            self.reset_ids: list[str] = []

        def get_active_flow_id(self) -> str | None:
            return self.active

        def flow_summary(self, flow_id: str | None = None):
            if not (flow_id or self.active):
                return None
            return {
                "flow_id": flow_id or self.active,
                "objective": "Implement parser retry flow",
                "constraints": ["Only touch src/ and tests/"],
                "open_tasks": ["add regression test"],
                "last_changed_files": ["src/mana_analyzer/services/ask_service.py"],
            }

        def checkpoint_flow(self, flow_id: str | None = None) -> str | None:
            target = flow_id or self.active
            if not target:
                return None
            self.checkpointed.append(target)
            return target

        def reset_flow(self, flow_id: str | None = None) -> str | None:
            target = flow_id or self.active
            if not target:
                return None
            self.reset_ids.append(target)
            self.active = None
            return target

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate(self, *_args: object, **_kwargs: object) -> dict:
            return {"answer": "ok", "changed_files": [], "warnings": [], "diff": "", "flow_id": self.active}

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:
            return {"answer": "ok", "changed_files": [], "warnings": [], "diff": "", "flow_id": self.active}

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_analyzer.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent", "--flow-id", "flow-123"],
        input="/flow show\n/flow checkpoint\n/flow reset\nquit\n",
    )
    assert result.exit_code == 0
    assert "Flow memory active" in result.stdout
    assert "Implement parser retry flow" in result.stdout


def test_chat_coding_agent_uses_worker_lifecycle_once(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeWorkerClient:
        start_calls = 0
        stop_calls = 0
        health_calls = 0

        def __init__(self, **_kwargs: object) -> None:
            return None

        def start(self) -> None:
            _FakeWorkerClient.start_calls += 1

        def health(self) -> dict[str, str]:
            _FakeWorkerClient.health_calls += 1
            return {"status": "ok"}

        def stop(self) -> None:
            _FakeWorkerClient.stop_calls += 1

    class _FakeCodingAgent:
        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-xyz"

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate(self, *_args: object, **_kwargs: object) -> dict:
            return {"answer": "ok", "changed_files": [], "warnings": [], "diff": "", "flow_id": self.active}

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:
            return {"answer": "ok", "changed_files": [], "warnings": [], "diff": "", "flow_id": self.active}

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_analyzer.commands.cli.ToolWorkerClient", _FakeWorkerClient)
    monkeypatch.setattr("mana_analyzer.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent"],
        input="please edit file\nanother edit\nquit\n",
    )
    assert result.exit_code == 0
    assert _FakeWorkerClient.start_calls == 1
    assert _FakeWorkerClient.health_calls == 1
    assert _FakeWorkerClient.stop_calls == 1


def test_flow_checklist_cli_view_renders_codex_sections(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-123"

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def flow_summary(self, flow_id: str | None = None):
            _ = flow_id
            return {
                "flow_id": "flow-123",
                "objective": "Implement planner",
                "checklist": {
                    "objective": "Implement planner",
                    "steps": [
                        {"status": "in_progress", "title": "Inspect file"},
                        {"status": "pending", "title": "Apply patch"},
                    ],
                },
            }

        def checkpoint_flow(self, flow_id: str | None = None) -> str | None:
            return flow_id or self.active

        def reset_flow(self, flow_id: str | None = None) -> str | None:
            _ = flow_id
            return self.active

        def generate(self, *_args: object, **_kwargs: object) -> dict:
            return {
                "answer": "done",
                "changed_files": ["src/a.py"],
                "warnings": [],
                "diff": "",
                "flow_id": self.active,
                "plan": {"objective": "Implement planner", "steps": [{"status": "done", "title": "Inspect file"}]},
                "progress": {"phase": "edit", "why": "gate passed", "budgets": {"search_used": 1, "search_budget": 4, "read_used": 2, "read_budget": 6, "read_files_observed": 2, "required_read_files": 2}},
                "checklist": {"done": 1, "pending": 0, "blocked": 0, "total": 1},
                "actions_taken": [{"tool_name": "read_file", "status": "ok", "duration_ms": 1.0, "args_summary": "x"}],
                "next_step": "Run verification",
                "static_analysis": {"finding_count": 0, "findings": []},
            }

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate()

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_analyzer.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent"],
        input="implement planner\n/flow checklist\nquit\n",
    )
    assert result.exit_code == 0
    assert "Plan" in result.stdout
    assert "Progress" in result.stdout
    assert "Checklist" in result.stdout
    assert "Steps" in result.stdout
    assert "Decisions" in result.stdout
    assert "History" in result.stdout
    assert "Next Step" in result.stdout
    assert "Flow Checklist" in result.stdout


def test_chat_coding_agent_answer_only_on_tools_only_fallback(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-123"

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate(self, *_args: object, **_kwargs: object) -> dict:
            return {
                "answer": "Request blocked by tools-only worker policy.",
                "changed_files": [],
                "warnings": ["tools_only_violation: no successful tool calls"],
                "diff": "",
                "flow_id": self.active,
                "actions_taken": [],
                "actions_taken_total": 0,
                "render_mode": "answer_only",
                "fallback_reason": "tools_only_violation",
                "fallback_retry_attempted": True,
            }

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate()

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_analyzer.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent"],
        input="update readme.md with new version\nquit\n",
    )
    assert result.exit_code == 0
    assert "Answer" in result.stdout
    assert "Request blocked by tools-only worker policy." in result.stdout
    assert "Summary" not in result.stdout
    assert "Steps" not in result.stdout
    assert "History" not in result.stdout
    assert "Next Step" not in result.stdout


def test_chat_coding_agent_answer_only_when_no_repo_edits(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-123"

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate(self, *_args: object, **_kwargs: object) -> dict:
            return {
                "answer": "Analysis complete. No repository edits required.",
                "changed_files": [],
                "warnings": [],
                "diff": "",
                "flow_id": self.active,
                "actions_taken": [
                    {
                        "tool_name": "read_file",
                        "status": "ok",
                        "duration_ms": 1.2,
                        "args_summary": "path='README.md' start=1 end=50",
                    }
                ],
                "actions_taken_total": 1,
                "plan": {"objective": "should not render", "steps": []},
                "progress": {"phase": "inspect", "why": "No edits needed"},
                "checklist": {"done": 1, "pending": 0, "blocked": 0, "total": 1},
                "next_step": "Done",
            }

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate()

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_analyzer.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent"],
        input="analyze this src,give me best plan for upgrade\nquit\n",
    )
    assert result.exit_code == 0
    assert "Answer" in result.stdout
    assert "Analysis complete. No repository edits required." in result.stdout
    assert "Summary" not in result.stdout
    assert "Steps" not in result.stdout
    assert "History" not in result.stdout
    assert "Plan" not in result.stdout
    assert "Checklist" not in result.stdout
    assert "Next Step" not in result.stdout


def test_large_json_answer_is_rendered_as_sections_not_raw_blob(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-123"

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate(self, *_args: object, **_kwargs: object) -> dict:
            huge = '{"answer":"' + ("x" * 5000) + '"}'
            return {
                "answer": huge,
                "changed_files": [],
                "warnings": [],
                "diff": "",
                "flow_id": self.active,
                "plan": {"objective": "obj", "steps": []},
                "progress": {"phase": "inspect", "why": "insufficient reads", "budgets": {"search_used": 2, "search_budget": 4, "read_used": 1, "read_budget": 6, "read_files_observed": 1, "required_read_files": 2}},
                "checklist": {"done": 0, "pending": 2, "blocked": 0, "total": 2},
                "actions_taken": [],
                "next_step": "Read one more file",
                "static_analysis": {"finding_count": 0, "findings": []},
            }

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate()

        def flow_summary(self, flow_id: str | None = None):
            _ = flow_id
            return None

        def checkpoint_flow(self, flow_id: str | None = None) -> str | None:
            return flow_id

        def reset_flow(self, flow_id: str | None = None) -> str | None:
            return flow_id

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_analyzer.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent"],
        input="implement\nquit\n",
    )
    assert result.exit_code == 0
    assert "Plan" in result.stdout
    assert "Next Step" in result.stdout


def test_chat_coding_agent_unlimited_mode_bypasses_default_step_cap(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        last_max_steps: int | None = None

        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-123"

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate(self, *_args: object, **kwargs: object) -> dict:
            _FakeCodingAgent.last_max_steps = int(kwargs.get("max_steps", 0) or 0)
            return {
                "answer": "done",
                "changed_files": [],
                "warnings": [],
                "diff": "",
                "flow_id": self.active,
                "plan": {"objective": "obj", "steps": []},
                "progress": {"phase": "edit", "why": "ok", "budgets": {"search_used": 0, "search_budget": 4, "read_used": 0, "read_budget": 6, "read_files_observed": 0, "required_read_files": 2}},
                "checklist": {"done": 0, "pending": 0, "blocked": 0, "total": 0},
                "actions_taken": [],
                "next_step": "done",
                "static_analysis": {"finding_count": 0, "findings": []},
            }

        def generate_dir_mode(self, *_args: object, **kwargs: object) -> dict:
            return self.generate(*_args, **kwargs)

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_analyzer.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent", "--agent-unlimited"],
        input="implement change\nquit\n",
    )
    assert result.exit_code == 0
    assert isinstance(_FakeCodingAgent.last_max_steps, int)
    assert _FakeCodingAgent.last_max_steps is not None
    assert _FakeCodingAgent.last_max_steps > 200


def test_chat_summary_uses_actions_taken_total_when_trace_is_truncated(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-123"

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate(self, *_args: object, **_kwargs: object) -> dict:
            return {
                "answer": "done",
                "changed_files": [],
                "warnings": [],
                "diff": "",
                "flow_id": self.active,
                "plan": {"objective": "obj", "steps": []},
                "progress": {"phase": "edit", "why": "ok", "budgets": {"search_used": 0, "search_budget": 4, "read_used": 0, "read_budget": 6, "read_files_observed": 0, "required_read_files": 2}},
                "checklist": {"done": 0, "pending": 0, "blocked": 0, "total": 0},
                "actions_taken_total": 37,
                "actions_taken_truncated": True,
                "actions_taken": [{"tool_name": "read_file", "status": "ok", "duration_ms": 1.0, "args_summary": "x"}],
                "next_step": "done",
                "static_analysis": {"finding_count": 0, "findings": []},
            }

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate(*_args, **_kwargs)

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_analyzer.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent"],
        input="implement\nquit\n",
    )
    assert result.exit_code == 0
    assert "- tool steps: 37" in result.stdout


def test_chat_renders_dynamic_plan_and_diagram_blocks_in_normal_path(monkeypatch, tmp_path: Path) -> None:
    class _DynamicAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

        def ask_with_tools(
            self,
            index_dir: str,
            question: str,
            k: int,
            max_steps: int = 6,
            timeout_seconds: int = 30,
        ) -> AskResponseWithTrace:
            _ = (index_dir, question, k, max_steps, timeout_seconds)
            payload = {
                "answer": "Dynamic answer",
                "ui_blocks": [
                    {
                        "type": "plan",
                        "title": "Architecture Plan",
                        "objective": "Ship dynamic UI",
                        "steps": [{"status": "in_progress", "title": "Render ui_blocks", "detail": "Use rich panel tables"}],
                    },
                    {
                        "type": "diagram",
                        "title": "Flow Diagram",
                        "format": "mermaid",
                        "content": "graph TD\nA-->B",
                    },
                ],
            }
            hit = SearchHit(0.9, "/tmp/good.py", 1, 3, "add", "snippet")
            return AskResponseWithTrace(
                answer=json.dumps(payload),
                sources=[hit],
                mode="agent-tools",
                trace=[],
                warnings=[],
            )

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_ask_service", lambda _s, model_override=None: _DynamicAskService())

    result = runner.invoke(
        app,
        ["chat", "--agent-tools"],
        input="show dynamic\nquit\n",
    )
    assert result.exit_code == 0
    assert "Architecture Plan" in result.stdout
    assert "Render ui_blocks" in result.stdout
    assert "Flow Diagram" in result.stdout
    assert "graph TD" in result.stdout


def test_chat_inferrs_mermaid_diagram_block_and_renders_before_summary(monkeypatch, tmp_path: Path) -> None:
    class _MermaidAskService(FakeAskService):
        calls: list[str] = []

        def __init__(self) -> None:
            self.ask_agent = object()
            _MermaidAskService.calls = []

        def ask_with_tools(
            self,
            index_dir: str,
            question: str,
            k: int,
            max_steps: int = 6,
            timeout_seconds: int = 30,
        ) -> AskResponseWithTrace:
            _ = (index_dir, k, max_steps, timeout_seconds)
            _MermaidAskService.calls.append(question)
            answer = "```mermaid\ngraph TD\nA-->B\n```"
            return AskResponseWithTrace(answer=answer, sources=[], mode="agent-tools", trace=[], warnings=[])

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_ask_service", lambda _s, model_override=None: _MermaidAskService())

    result = runner.invoke(
        app,
        ["chat", "--agent-tools"],
        input="show diagram\nquit\n",
    )
    assert result.exit_code == 0
    assert len(_MermaidAskService.calls) == 1
    assert "graph TD" in result.stdout
    assert result.stdout.find("Diagram") < result.stdout.find("Summary")


def test_chat_diagram_artifact_render_invokes_mermaid_renderer(monkeypatch, tmp_path: Path) -> None:
    class _DiagramAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

        def ask_with_tools(
            self,
            index_dir: str,
            question: str,
            k: int,
            max_steps: int = 6,
            timeout_seconds: int = 30,
        ) -> AskResponseWithTrace:
            _ = (index_dir, question, k, max_steps, timeout_seconds)
            payload = {
                "answer": "diagram answer",
                "ui_blocks": [
                    {
                        "type": "diagram",
                        "title": "Flow Diagram",
                        "format": "mermaid",
                        "content": "graph TD\nA-->B",
                    }
                ],
            }
            return AskResponseWithTrace(answer=json.dumps(payload), sources=[], mode="agent-tools", trace=[], warnings=[])

    calls: list[dict] = []

    def _fake_render_mermaid_artifact(
        content: str,
        *,
        output_dir: Path,
        title: str,
        image_format: str,
        timeout_seconds: int,
        project_root: Path | None = None,
    ):
        calls.append(
            {
                "content": content,
                "output_dir": output_dir,
                "title": title,
                "image_format": image_format,
                "timeout_seconds": timeout_seconds,
                "project_root": project_root,
            }
        )
        return (tmp_path / "flow.svg", None)

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_ask_service", lambda _s, model_override=None: _DiagramAskService())
    monkeypatch.setattr("mana_analyzer.commands.cli._render_mermaid_artifact", _fake_render_mermaid_artifact)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--diagram-output-dir", str(tmp_path), "--diagram-format", "svg"],
        input="show diagram\nquit\n",
    )
    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0]["content"] == "graph TD\nA-->B"
    assert calls[0]["image_format"] == "svg"
    assert "Diagram Artifact" in result.stdout
    assert str(tmp_path / "flow.svg") in result.stdout


def test_chat_no_diagram_render_images_skips_mermaid_artifact(monkeypatch, tmp_path: Path) -> None:
    class _DiagramAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

        def ask_with_tools(
            self,
            index_dir: str,
            question: str,
            k: int,
            max_steps: int = 6,
            timeout_seconds: int = 30,
        ) -> AskResponseWithTrace:
            _ = (index_dir, question, k, max_steps, timeout_seconds)
            payload = {
                "answer": "diagram answer",
                "ui_blocks": [
                    {
                        "type": "diagram",
                        "title": "Flow Diagram",
                        "format": "mermaid",
                        "content": "graph TD\nA-->B",
                    }
                ],
            }
            return AskResponseWithTrace(answer=json.dumps(payload), sources=[], mode="agent-tools", trace=[], warnings=[])

    calls: list[dict] = []

    def _fake_render_mermaid_artifact(
        content: str,
        *,
        output_dir: Path,
        title: str,
        image_format: str,
        timeout_seconds: int,
        project_root: Path | None = None,
    ):
        _ = (content, output_dir, title, image_format, timeout_seconds, project_root)
        calls.append({})
        return (tmp_path / "flow.svg", None)

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_ask_service", lambda _s, model_override=None: _DiagramAskService())
    monkeypatch.setattr("mana_analyzer.commands.cli._render_mermaid_artifact", _fake_render_mermaid_artifact)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--no-diagram-render-images"],
        input="show diagram\nquit\n",
    )
    assert result.exit_code == 0
    assert len(calls) == 0
    assert "Diagram Artifact" not in result.stdout


def test_chat_coding_path_prefers_dynamic_plan_over_static_plan_section(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-123"

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate(self, *_args: object, **_kwargs: object) -> dict:
            payload = {
                "answer": "dynamic",
                "ui_blocks": [
                    {
                        "type": "plan",
                        "title": "Dynamic Plan",
                        "objective": "DYNAMIC_OBJECTIVE",
                        "steps": [{"status": "done", "title": "done step"}],
                    }
                ],
            }
            return {
                "answer": json.dumps(payload),
                "changed_files": [],
                "warnings": [],
                "diff": "",
                "flow_id": self.active,
                "plan": {"objective": "STATIC_PLAN_SHOULD_NOT_RENDER", "steps": [{"status": "pending", "title": "static"}]},
                "progress": {"phase": "edit", "why": "ok", "budgets": {"search_used": 0, "search_budget": 4, "read_used": 0, "read_budget": 6, "read_files_observed": 0, "required_read_files": 2}},
                "checklist": {"done": 0, "pending": 0, "blocked": 0, "total": 0},
                "actions_taken": [],
                "next_step": "done",
                "static_analysis": {"finding_count": 0, "findings": []},
            }

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate(*_args, **_kwargs)

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_analyzer.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent"],
        input="implement\nquit\n",
    )
    assert result.exit_code == 0
    assert "DYNAMIC_OBJECTIVE" in result.stdout
    assert "STATIC_PLAN_SHOULD_NOT_RENDER" not in result.stdout


def test_chat_coding_path_inferrs_mermaid_diagram_block_and_renders_before_summary(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-123"

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate(self, *_args: object, **_kwargs: object) -> dict:
            return {
                "answer": "```mermaid\ngraph LR\nX-->Y\n```",
                "changed_files": [],
                "warnings": [],
                "diff": "",
                "flow_id": self.active,
                "plan": {"objective": "obj", "steps": []},
                "progress": {"phase": "edit", "why": "ok", "budgets": {"search_used": 0, "search_budget": 4, "read_used": 0, "read_budget": 6, "read_files_observed": 0, "required_read_files": 2}},
                "checklist": {"done": 0, "pending": 0, "blocked": 0, "total": 0},
                "actions_taken": [],
                "next_step": "done",
                "static_analysis": {"finding_count": 0, "findings": []},
            }

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate(*_args, **_kwargs)

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_analyzer.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent"],
        input="diagram\nquit\n",
    )
    assert result.exit_code == 0
    assert "graph LR" in result.stdout
    assert result.stdout.find("Diagram") < result.stdout.find("Summary")


def test_chat_ignores_malformed_ui_blocks_and_falls_back_to_answer(monkeypatch, tmp_path: Path) -> None:
    class _MalformedAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

        def ask_with_tools(
            self,
            index_dir: str,
            question: str,
            k: int,
            max_steps: int = 6,
            timeout_seconds: int = 30,
        ) -> AskResponseWithTrace:
            _ = (index_dir, question, k, max_steps, timeout_seconds)
            payload = {
                "answer": "Fallback answer",
                "ui_blocks": [
                    {"type": "diagram", "content": ""},
                    {"no_type": "x"},
                    "invalid",
                ],
            }
            return AskResponseWithTrace(answer=json.dumps(payload), sources=[], mode="agent-tools", trace=[], warnings=[])

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_ask_service", lambda _s, model_override=None: _MalformedAskService())

    result = runner.invoke(
        app,
        ["chat", "--agent-tools"],
        input="hello\nquit\n",
    )
    assert result.exit_code == 0
    assert "Fallback answer" in result.stdout


def test_chat_selection_flow_accepts_numeric_choice_and_synthesizes_follow_up(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        calls: list[str] = []

        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-123"
            _FakeCodingAgent.calls = []

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def _base_result(self, answer: str) -> dict:
            return {
                "answer": answer,
                "changed_files": [],
                "warnings": [],
                "diff": "",
                "flow_id": self.active,
                "plan": {"objective": "obj", "steps": []},
                "progress": {"phase": "inspect", "why": "ok", "budgets": {"search_used": 0, "search_budget": 4, "read_used": 0, "read_budget": 6, "read_files_observed": 0, "required_read_files": 2}},
                "checklist": {"done": 0, "pending": 0, "blocked": 0, "total": 0},
                "actions_taken": [],
                "next_step": "done",
                "static_analysis": {"finding_count": 0, "findings": []},
            }

        def generate(self, question: str, *_args: object, **_kwargs: object) -> dict:
            _FakeCodingAgent.calls.append(question)
            if len(_FakeCodingAgent.calls) == 1:
                payload = {
                    "answer": "",
                    "ui_blocks": [
                        {
                            "type": "selection",
                            "id": "mode_select",
                            "prompt": "Pick a mode",
                            "options": [
                                {"id": "safe", "label": "Safe mode", "value": "safe"},
                                {"id": "fast", "label": "Fast mode", "value": "speed"},
                            ],
                        }
                    ],
                }
                return self._base_result(json.dumps(payload))
            return self._base_result("Selection applied")

        def generate_dir_mode(self, *args: object, **kwargs: object) -> dict:
            return self.generate(*args, **kwargs)

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_analyzer.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent"],
        input="begin\n2\nquit\n",
    )
    assert result.exit_code == 0
    assert len(_FakeCodingAgent.calls) == 2
    assert _FakeCodingAgent.calls[1] == (
        'User selected "fast" for selection "mode_select" (value="speed"). Continue accordingly.'
    )


def test_chat_selection_flow_reprompts_on_invalid_choice(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        calls: list[str] = []

        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-123"
            _FakeCodingAgent.calls = []

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def _base_result(self, answer: str) -> dict:
            return {
                "answer": answer,
                "changed_files": [],
                "warnings": [],
                "diff": "",
                "flow_id": self.active,
                "plan": {"objective": "obj", "steps": []},
                "progress": {"phase": "inspect", "why": "ok", "budgets": {"search_used": 0, "search_budget": 4, "read_used": 0, "read_budget": 6, "read_files_observed": 0, "required_read_files": 2}},
                "checklist": {"done": 0, "pending": 0, "blocked": 0, "total": 0},
                "actions_taken": [],
                "next_step": "done",
                "static_analysis": {"finding_count": 0, "findings": []},
            }

        def generate(self, question: str, *_args: object, **_kwargs: object) -> dict:
            _FakeCodingAgent.calls.append(question)
            if len(_FakeCodingAgent.calls) == 1:
                payload = {
                    "answer": "",
                    "ui_blocks": [
                        {
                            "type": "continue",
                            "prompt": "Continue current flow?",
                            "options": [
                                {"id": "continue", "label": "Continue"},
                                {"id": "new", "label": "Start new"},
                            ],
                        }
                    ],
                }
                return self._base_result(json.dumps(payload))
            return self._base_result("Handled")

        def generate_dir_mode(self, *args: object, **kwargs: object) -> dict:
            return self.generate(*args, **kwargs)

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_analyzer.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent"],
        input="begin\nbad choice\nnew\nquit\n",
    )
    assert result.exit_code == 0
    assert "Invalid selection" in result.stdout
    assert len(_FakeCodingAgent.calls) == 2
    assert _FakeCodingAgent.calls[1] == (
        'User selected "new" for selection "continue_selection" (value="new"). Continue accordingly.'
    )


def test_chat_selection_flow_works_in_normal_agent_tools_path(monkeypatch, tmp_path: Path) -> None:
    class _SelectionAskService(FakeAskService):
        calls: list[str] = []

        def __init__(self) -> None:
            self.ask_agent = object()
            _SelectionAskService.calls = []

        def ask_with_tools(
            self,
            index_dir: str,
            question: str,
            k: int,
            max_steps: int = 6,
            timeout_seconds: int = 30,
        ) -> AskResponseWithTrace:
            _ = (index_dir, k, max_steps, timeout_seconds)
            _SelectionAskService.calls.append(question)
            if len(_SelectionAskService.calls) == 1:
                payload = {
                    "answer": "",
                    "ui_blocks": [
                        {
                            "type": "selection",
                            "id": "normal_select",
                            "prompt": "Pick one",
                            "options": [
                                {"id": "one", "label": "One"},
                                {"id": "two", "label": "Two"},
                            ],
                        }
                    ],
                }
                return AskResponseWithTrace(answer=json.dumps(payload), sources=[], mode="agent-tools", trace=[], warnings=[])
            hit_a = SearchHit(0.8, "/tmp/a.py", 1, 2, "a", "snippet")
            hit_b = SearchHit(0.8, "/tmp/b.py", 3, 4, "b", "snippet")
            return AskResponseWithTrace(
                answer="Normal path handled",
                sources=[hit_a, hit_b],
                mode="agent-tools",
                trace=[],
                warnings=[],
            )

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_ask_service", lambda _s, model_override=None: _SelectionAskService())

    result = runner.invoke(
        app,
        ["chat", "--agent-tools"],
        input="begin\n2\nquit\n",
    )
    assert result.exit_code == 0
    assert len(_SelectionAskService.calls) == 2
    assert _SelectionAskService.calls[1] == (
        'User selected "two" for selection "normal_select" (value="two"). Continue accordingly.'
    )
