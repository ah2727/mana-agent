from __future__ import annotations

import json
from pathlib import Path

from mana_analyzer.llm.coding_agent import CodingAgent, FlowChecklist, FlowStep
from mana_analyzer.llm.tool_worker_process import ToolWorkerProcessError
from mana_analyzer.services.coding_memory_service import CodingMemoryService


class _Tool:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeAskAgent:
    def __init__(self, response_payload: dict) -> None:
        self.tools: list[object] = []
        self.model = "fake"
        self.calls: list[dict] = []
        self.response_payload = response_payload

    def run(
        self,
        question: str,
        index_dir: str | Path,
        k: int,
        max_steps: int,
        timeout_seconds: int,
        callbacks: list[object] | None = None,
        system_prompt: str | None = None,
        tool_policy: dict | None = None,
    ) -> str:
        _ = (index_dir, k, max_steps, timeout_seconds, callbacks, system_prompt)
        self.calls.append({"question": question, "tool_policy": tool_policy or {}})
        return json.dumps(self.response_payload)

    def run_multi(
        self,
        question: str,
        index_dirs,
        k: int,
        max_steps: int,
        timeout_seconds: int,
        callbacks: list[object] | None = None,
        system_prompt: str | None = None,
        tool_policy: dict | None = None,
    ) -> str:
        _ = (index_dirs, k, max_steps, timeout_seconds, callbacks, system_prompt)
        self.calls.append({"question": question, "tool_policy": tool_policy or {}})
        return json.dumps(self.response_payload)


def _fixed_checklist() -> FlowChecklist:
    return FlowChecklist(
        objective="Implement request",
        constraints=["scope src/ tests/ only"],
        acceptance=["tests pass"],
        steps=[
            FlowStep(id="s1", title="Discover relevant files", reason="find exact code", status="in_progress"),
            FlowStep(id="s2", title="Inspect file contents", reason="validate behavior", status="pending"),
            FlowStep(id="s3", title="Apply edit", reason="implement request", status="pending"),
        ],
        next_action="Inspect target files.",
    )


def _build_agent(
    tmp_path: Path,
    monkeypatch,
    *,
    payload: dict,
    memory: bool = True,
) -> CodingAgent:
    monkeypatch.setattr("mana_analyzer.llm.coding_agent.build_write_file_tool", lambda **_kwargs: _Tool("write_file"))
    monkeypatch.setattr("mana_analyzer.llm.coding_agent.build_apply_patch_tool", lambda **_kwargs: _Tool("apply_patch"))
    ask_agent = _FakeAskAgent(payload)
    svc = CodingMemoryService(project_root=tmp_path, max_turns=5, max_tasks=20) if memory else None
    agent = CodingAgent(
        api_key="test-key",
        repo_root=tmp_path,
        ask_agent=ask_agent,
        allowed_prefixes=None,
        coding_memory_service=svc,
        coding_memory_enabled=memory,
        plan_max_steps=8,
        search_budget=4,
        read_budget=6,
        require_read_files=2,
    )
    monkeypatch.setattr(agent, "_plan_checklist", lambda request, flow_context=None: (_fixed_checklist(), []))
    monkeypatch.setattr(agent, "_git_status_paths", lambda: set())  # type: ignore[method-assign]
    monkeypatch.setattr(agent, "_git_diff", lambda _paths: "")  # type: ignore[method-assign]
    monkeypatch.setattr(agent, "_run_static_analysis", lambda _paths: [])  # type: ignore[method-assign]
    return agent


def test_coding_agent_builds_structured_checklist_before_tools(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    result = agent.generate("Implement planner", index_dir=tmp_path / ".mana_index", k=4)
    assert isinstance(result.get("plan"), dict)
    assert result["plan"]["objective"] == "Implement request"
    assert result["checklist"]["total"] >= 1


def test_coding_agent_blocks_answer_until_required_file_reads_met(tmp_path: Path, monkeypatch) -> None:
    payload = {
        "answer": "ok",
        "trace": [
            {"tool_name": "semantic_search", "status": "ok", "duration_ms": 2.0, "args_summary": "q=planner"},
            {"tool_name": "read_file", "status": "ok", "duration_ms": 3.0, "args_summary": "one"},
        ],
        "warnings": [],
    }
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    result = agent.generate("Implement planner", index_dir=tmp_path / ".mana_index", k=4)
    assert result["progress"]["phase"] == "blocked"
    assert "Need at least 2 unique read_file inspections" in result["progress"]["why"]


def test_coding_agent_prevents_duplicate_semantic_search_loops(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    result = agent.generate("Implement planner", index_dir=tmp_path / ".mana_index", k=4)
    fake = agent.ask_agent
    assert isinstance(fake, _FakeAskAgent)
    assert fake.calls
    policy = fake.calls[0]["tool_policy"]
    assert policy["search_repeat_limit"] == 1


def test_coding_agent_enforces_search_budget_and_transitions_phase(tmp_path: Path, monkeypatch) -> None:
    payload = {
        "answer": "ok",
        "trace": [
            {"tool_name": "semantic_search", "status": "ok", "duration_ms": 1.0, "args_summary": "a"},
            {"tool_name": "semantic_search", "status": "ok", "duration_ms": 1.0, "args_summary": "b"},
            {"tool_name": "read_file", "status": "ok", "duration_ms": 1.0, "args_summary": "file1"},
            {"tool_name": "read_file", "status": "ok", "duration_ms": 1.0, "args_summary": "file2"},
        ],
        "warnings": [],
    }
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    result = agent.generate("Implement planner", index_dir=tmp_path / ".mana_index", k=4)
    fake = agent.ask_agent
    assert isinstance(fake, _FakeAskAgent)
    policy = fake.calls[0]["tool_policy"]
    assert policy["search_budget"] == 4
    assert result["progress"]["budgets"]["search_used"] == 2


def test_coding_agent_repo_only_default_disables_internet_without_explicit_user_request(
    tmp_path: Path, monkeypatch
) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    agent.generate("Implement parser fix", index_dir=tmp_path / ".mana_index", k=4)
    fake = agent.ask_agent
    assert isinstance(fake, _FakeAskAgent)
    assert fake.calls[0]["tool_policy"]["block_internet"] is True
    agent.generate("Need latest internet docs for this API", index_dir=tmp_path / ".mana_index", k=4)
    assert fake.calls[1]["tool_policy"]["block_internet"] is False


def test_flow_checklist_persists_and_resumes_across_turns(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    first = agent.generate("Implement A", index_dir=tmp_path / ".mana_index", k=4)
    second = agent.generate("Continue A", index_dir=tmp_path / ".mana_index", k=4, flow_id=first["flow_id"])
    assert isinstance(first.get("flow_id"), str)
    assert second["flow_id"] == first["flow_id"]
    summary = agent.flow_summary(first["flow_id"])
    assert isinstance(summary, dict)
    assert isinstance(summary.get("checklist"), dict)


def test_dir_mode_coding_agent_flow_context_included(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    first = agent.generate_dir_mode(
        "Implement dir-mode flow",
        index_dirs=[tmp_path / "proj/.mana_index"],
        k=4,
    )
    assert isinstance(first.get("plan"), dict)


def test_coding_agent_handles_tools_only_worker_violation(tmp_path: Path, monkeypatch) -> None:
    class _FailingWorker:
        def run_tools(self, _request):
            raise ToolWorkerProcessError(
                code="tools_only_violation",
                message="no successful tool calls",
                retriable=False,
            )

    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    agent.tool_worker_client = _FailingWorker()
    result = agent.generate("Implement planner", index_dir=tmp_path / ".mana_index", k=4)
    assert "tools-only worker policy" in str(result.get("answer", "")).lower()
    warnings = result.get("warnings") or []
    assert any("tools_only_violation" in str(item) for item in warnings)
