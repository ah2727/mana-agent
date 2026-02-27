from __future__ import annotations

import json
from pathlib import Path

from mana_analyzer.llm.coding_agent import CodingAgent, FlowChecklist, FlowStep
from mana_analyzer.llm.tool_worker_process import ToolWorkerProcessError
from mana_analyzer.llm.tools_manager import AutoExecuteResult
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


class _SequenceAskAgent(_FakeAskAgent):
    def __init__(self, response_payloads: list[dict]) -> None:
        super().__init__(response_payload=response_payloads[0] if response_payloads else {})
        self._payloads = list(response_payloads)
        self._cursor = 0

    def _next_payload(self) -> dict:
        if not self._payloads:
            return {}
        if self._cursor >= len(self._payloads):
            return self._payloads[-1]
        payload = self._payloads[self._cursor]
        self._cursor += 1
        return payload

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
        return json.dumps(self._next_payload())

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
        return json.dumps(self._next_payload())


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


def _build_agent_with_ask(tmp_path: Path, monkeypatch, ask_agent) -> CodingAgent:
    monkeypatch.setattr("mana_analyzer.llm.coding_agent.build_write_file_tool", lambda **_kwargs: _Tool("write_file"))
    monkeypatch.setattr("mana_analyzer.llm.coding_agent.build_apply_patch_tool", lambda **_kwargs: _Tool("apply_patch"))
    svc = CodingMemoryService(project_root=tmp_path, max_turns=5, max_tasks=20)
    agent = CodingAgent(
        api_key="test-key",
        repo_root=tmp_path,
        ask_agent=ask_agent,
        allowed_prefixes=None,
        coding_memory_service=svc,
        coding_memory_enabled=True,
        plan_max_steps=8,
        search_budget=4,
        read_budget=6,
        require_read_files=1,
    )
    monkeypatch.setattr(agent, "_plan_checklist", lambda request, flow_context=None: (_fixed_checklist(), []))
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


def test_coding_agent_effective_prompt_includes_language_tooling_guide(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    prompt = agent._effective_system_prompt_for("Fix failing pytest and npm test flows")
    lowered = prompt.lower()
    assert ".venv" in lowered
    assert "node_modules" in lowered
    assert "npm install" in lowered
    assert "pytest -q" in lowered


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


def test_dir_mode_rewrites_ambiguous_followup_with_flow_context(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    first = agent.generate_dir_mode(
        "Refactor parser and update prompts",
        index_dirs=[tmp_path / "proj/.mana_index"],
        k=4,
    )
    flow_id = str(first.get("flow_id") or "")
    second = agent.generate_dir_mode(
        "yes.",
        index_dirs=[tmp_path / "proj/.mana_index"],
        k=4,
        flow_id=flow_id,
    )
    _ = second

    fake = agent.ask_agent
    assert isinstance(fake, _FakeAskAgent)
    assert len(fake.calls) >= 2
    followup_question = str(fake.calls[1]["question"])
    assert followup_question != "yes."
    assert "Current objective:" in followup_question
    warnings = second.get("warnings") or []
    assert any("followup_request_rewritten_from_flow_context" in str(item) for item in warnings)


def test_plan_trigger_followup_rewrites_to_execute_checklist(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    first = agent.generate(
        "Update docs and type hints in TODO.md",
        index_dir=tmp_path / ".mana_index",
        k=4,
    )
    flow_id = str(first.get("flow_id") or "")
    second = agent.generate(
        "implement plan.",
        index_dir=tmp_path / ".mana_index",
        k=4,
        flow_id=flow_id,
    )

    fake = agent.ask_agent
    assert isinstance(fake, _FakeAskAgent)
    assert len(fake.calls) >= 2
    followup_question = str(fake.calls[1]["question"])
    assert "Execute the active flow checklist now." in followup_question
    assert "Current objective:" in followup_question
    warnings = second.get("warnings") or []
    assert any("followup_request_rewritten_from_flow_context" in str(item) for item in warnings)


def test_plan_trigger_followup_not_classified_as_conflict(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    first = agent.generate(
        "Refactor coding agent flow prompts and docs",
        index_dir=tmp_path / ".mana_index",
        k=4,
    )
    flow_id = str(first.get("flow_id") or "")
    assert agent.is_conflicting_request("implement plan.", flow_id) is False


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


def test_coding_agent_retries_tools_only_violation_once_with_override(tmp_path: Path, monkeypatch) -> None:
    class _RetryWorker:
        def __init__(self) -> None:
            self.calls = 0
            self.requests = []

        def run_tools(self, request):
            self.calls += 1
            self.requests.append(request)
            if self.calls == 1:
                raise ToolWorkerProcessError(
                    code="tools_only_violation",
                    message="no successful tool calls",
                    retriable=False,
                )
            return type(
                "_Resp",
                (),
                {
                    "model_dump": lambda self: {
                        "answer": "fallback answer",
                        "sources": [],
                        "mode": "agent-tools",
                        "trace": [],
                        "warnings": [],
                    }
                },
            )()

    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    worker = _RetryWorker()
    agent.tool_worker_client = worker
    result = agent.generate("Implement planner", index_dir=tmp_path / ".mana_index", k=4)
    assert worker.calls == 2
    assert worker.requests[1].tools_only_strict_override is False
    assert result.get("render_mode") == "answer_only"
    assert result.get("fallback_reason") == "tools_only_violation"
    assert result.get("fallback_retry_attempted") is True


def test_coding_agent_force_fallback_retry_uses_retry_request_text(tmp_path: Path, monkeypatch) -> None:
    ask_agent = _SequenceAskAgent(
        [
            {
                "answer": "first",
                "trace": [
                    {"tool_name": "read_file", "status": "ok", "output_preview": '{"file_path":"README.md"}'},
                    {"tool_name": "apply_patch", "status": "error"},
                    {"tool_name": "apply_patch", "status": "error"},
                ],
                "warnings": [],
            }
        ]
    )
    agent = _build_agent_with_ask(tmp_path, monkeypatch, ask_agent)
    monkeypatch.setattr(agent, "_git_status_paths", lambda: set())  # type: ignore[method-assign]

    result = agent.generate("update README section", index_dir=tmp_path / ".mana_index", k=4)
    _ = result
    assert len(ask_agent.calls) >= 2
    assert any(
        "apply_patch is unavailable or failed/no-op repeatedly" in str(call.get("question", ""))
        for call in ask_agent.calls
    )
    warnings = result.get("warnings") or []
    assert any("forced_write_file_retry_after_patch_noop" in str(item) for item in warnings)


def test_coding_agent_noop_patch_then_write_change_avoids_blocked(tmp_path: Path, monkeypatch) -> None:
    ask_agent = _SequenceAskAgent(
        [
            {
                "answer": "first",
                "trace": [
                    {"tool_name": "read_file", "status": "ok", "output_preview": '{"file_path":"README.md"}'},
                    {"tool_name": "apply_patch", "status": "error"},
                    {"tool_name": "apply_patch", "status": "error"},
                ],
                "warnings": [],
            },
            {
                "answer": "write done",
                "trace": [
                    {"tool_name": "write_file", "status": "ok"},
                ],
                "warnings": [],
            },
        ]
    )
    agent = _build_agent_with_ask(tmp_path, monkeypatch, ask_agent)
    states = iter([set(), set(), {"README.md"}])
    monkeypatch.setattr(agent, "_git_status_paths", lambda: next(states, {"README.md"}))  # type: ignore[method-assign]

    result = agent.generate("update README section", index_dir=tmp_path / ".mana_index", k=4)
    assert result["progress"]["phase"] != "blocked"
    assert result["changed_files"] == ["README.md"]


def test_coding_agent_true_blocker_after_bounded_noop_retries(tmp_path: Path, monkeypatch) -> None:
    ask_agent = _SequenceAskAgent(
        [
            {
                "answer": "first",
                "trace": [
                    {"tool_name": "read_file", "status": "ok", "output_preview": '{"file_path":"README.md"}'},
                    {"tool_name": "apply_patch", "status": "error"},
                    {"tool_name": "apply_patch", "status": "error"},
                ],
                "warnings": [],
            },
            {
                "answer": "still no change",
                "trace": [
                    {"tool_name": "apply_patch", "status": "error"},
                    {"tool_name": "write_file", "status": "ok"},
                ],
                "warnings": [],
            },
        ]
    )
    agent = _build_agent_with_ask(tmp_path, monkeypatch, ask_agent)
    monkeypatch.setattr(agent, "_git_status_paths", lambda: set())  # type: ignore[method-assign]

    result = agent.generate("update README section", index_dir=tmp_path / ".mana_index", k=4)
    assert result["progress"]["phase"] == "blocked"
    assert "mutation_exhausted_true_blocker" in str(result["progress"]["why"])
    warnings = result.get("warnings") or []
    assert any("mutation_exhausted_true_blocker" in str(item) for item in warnings)


def test_coding_agent_conversational_noop_retries_before_terminal(tmp_path: Path, monkeypatch) -> None:
    ask_agent = _SequenceAskAgent(
        [
            {
                "answer": "If you want, I can proceed with edits.",
                "trace": [
                    {"tool_name": "read_file", "status": "ok", "output_preview": '{"file_path":"README.md"}'},
                ],
                "warnings": [],
            },
            {
                "answer": "completed edit",
                "trace": [
                    {"tool_name": "write_file", "status": "ok"},
                ],
                "warnings": [],
            },
        ]
    )
    agent = _build_agent_with_ask(tmp_path, monkeypatch, ask_agent)
    states = iter([set(), set(), {"README.md"}])
    monkeypatch.setattr(agent, "_git_status_paths", lambda: next(states, {"README.md"}))  # type: ignore[method-assign]

    result = agent.generate("please update readme.md", index_dir=tmp_path / ".mana_index", k=4)
    assert len(ask_agent.calls) >= 2
    assert result["changed_files"] == ["README.md"]
    warnings = result.get("warnings") or []
    assert any("edit_intent_conversational_noop_detected" in str(item) for item in warnings)


def test_plan_checklist_falls_back_when_planner_json_is_invalid(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    agent._plan_checklist = CodingAgent._plan_checklist.__get__(agent, CodingAgent)  # type: ignore[method-assign]

    class _BadPlanner:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            # first call: invalid schema text, second call: invalid repair text
            if self.calls == 1:
                return type("_Msg", (), {"content": "not json"})()
            return type("_Msg", (), {"content": "```json\n{ broken }\n```"})()

    agent.planner_llm = _BadPlanner()
    checklist, warnings = agent._plan_checklist("update README.md with new version")
    assert checklist is not None
    assert checklist.steps
    assert any("planner parse failed" in str(item) for item in warnings)


def test_generate_auto_execute_delegates_to_tools_manager_orchestrator(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)

    class _Orchestrator:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, **_kwargs):
            self.calls += 1
            return AutoExecuteResult(
                answer="auto-complete",
                trace=[{"tool_name": "read_file", "status": "ok", "duration_ms": 1.0, "args_summary": "x"}],
                warnings=[],
                changed_files=[],
                plan={"objective": "Implement request", "steps": []},
                passes=1,
                terminal_reason="manager_stop",
                toolsmanager_requests_count=1,
                pass_logs=[{"pass_index": 1, "requests_count": 1}],
            )

    orchestrator = _Orchestrator()
    agent.set_tools_manager_orchestrator(orchestrator)
    result = agent.generate_auto_execute(
        "Implement planner",
        index_dir=tmp_path / ".mana_index",
        k=4,
        pass_cap=3,
    )
    assert orchestrator.calls == 1
    assert result["answer"] == "auto-complete"
    assert result["auto_execute_passes"] == 1
    assert result["auto_execute_terminal_reason"] == "manager_stop"


def test_generate_auto_execute_preserves_flow_id_continuity(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)

    class _Orchestrator:
        def run(self, **_kwargs):
            return AutoExecuteResult(
                answer="auto",
                trace=[],
                warnings=[],
                changed_files=[],
                plan={"objective": "Implement request", "steps": []},
                passes=1,
                terminal_reason="manager_stop",
                toolsmanager_requests_count=1,
                pass_logs=[{"pass_index": 1, "requests_count": 1}],
            )

    agent.set_tools_manager_orchestrator(_Orchestrator())
    first = agent.generate_auto_execute("Implement A", index_dir=tmp_path / ".mana_index", k=4)
    second = agent.generate_auto_execute(
        "Continue A",
        index_dir=tmp_path / ".mana_index",
        k=4,
        flow_id=first["flow_id"],
    )
    assert isinstance(first.get("flow_id"), str)
    assert second.get("flow_id") == first.get("flow_id")


def test_plan_checklist_parses_python_literal_payload_without_repair(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    agent._plan_checklist = CodingAgent._plan_checklist.__get__(agent, CodingAgent)  # type: ignore[method-assign]

    class _LiteralPlanner:
        def invoke(self, _messages):
            return type(
                "_Msg",
                (),
                {
                    "content": (
                        "{'objective': 'Update README', 'constraints': [], 'acceptance': ['done'], "
                        "'steps': [{'id': 's1', 'title': 'Inspect README.md', 'reason': 'Gather context', "
                        "'status': 'in_progress', 'requires_tools': ['read_file']}], "
                        "'next_action': 'Inspect README.md'}"
                    )
                },
            )()

    agent.planner_llm = _LiteralPlanner()
    checklist, warnings = agent._plan_checklist("update README.md with new version")
    assert checklist is not None
    assert checklist.objective == "Update README"
    assert checklist.steps
    assert warnings == []


def test_plan_checklist_parses_list_blocks_with_plan_text(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    agent._plan_checklist = CodingAgent._plan_checklist.__get__(agent, CodingAgent)  # type: ignore[method-assign]

    class _ListPlanner:
        def invoke(self, _messages):
            return type(
                "_Msg",
                (),
                {
                    "content": (
                        "[{'id': 'rs_1', 'summary': [], 'type': 'reasoning'}, "
                        "{'type': 'text', 'text': 'Plan:\\n1. Inspect README.md\\n2. Update docs\\n3. Verify formatting'}]"
                    )
                },
            )()

    agent.planner_llm = _ListPlanner()
    checklist, warnings = agent._plan_checklist("update README.md with new version")
    assert checklist is not None
    assert len(checklist.steps) >= 3
    assert checklist.steps[0].title.startswith("Inspect README.md")
    assert warnings == []


def test_plan_checklist_parses_wrapped_answer_markdown_payload_without_repair(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    agent._plan_checklist = CodingAgent._plan_checklist.__get__(agent, CodingAgent)  # type: ignore[method-assign]

    class _WrappedAnswerPlanner:
        def invoke(self, _messages):
            wrapped = {
                "answer": (
                    "[{'id': 'rs_1', 'summary': [], 'type': 'reasoning'}, "
                    "{'type': 'text', 'text': '**Execution Plan**\\n"
                    "1. Inspect README.md\\n2. Update docs\\n3. Verify formatting'}]"
                )
            }
            return type("_Msg", (), {"content": json.dumps(wrapped)})()

    agent.planner_llm = _WrappedAnswerPlanner()
    checklist, warnings = agent._plan_checklist("update README.md with new version")
    assert checklist is not None
    assert len(checklist.steps) >= 3
    assert checklist.steps[0].title.startswith("Inspect README.md")
    assert warnings == []


def test_plan_checklist_parses_wrapped_content_markdown_fence_without_repair(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    agent._plan_checklist = CodingAgent._plan_checklist.__get__(agent, CodingAgent)  # type: ignore[method-assign]

    class _WrappedContentPlanner:
        def invoke(self, _messages):
            wrapped = {
                "content": (
                    "```markdown\n"
                    "**Execution Plan**\n"
                    "1. Inspect README.md\n"
                    "2. Apply targeted patch\n"
                    "3. Verify with tests\n"
                    "```"
                )
            }
            return type("_Msg", (), {"content": json.dumps(wrapped)})()

    agent.planner_llm = _WrappedContentPlanner()
    checklist, warnings = agent._plan_checklist("update README.md with new version")
    assert checklist is not None
    assert len(checklist.steps) >= 3
    assert checklist.steps[0].title.startswith("Inspect README.md")
    assert warnings == []


def test_parse_flow_checklist_json_uses_nested_answer_field() -> None:
    wrapped = {
        "answer": (
            "[{'id': 'rs_1', 'summary': [], 'type': 'reasoning'}, "
            "{'type': 'text', 'text': 'Execution Plan:\\n"
            "1. Inspect src/example.py\\n2. Apply change\\n3. Verify output'}]"
        )
    }
    checklist = CodingAgent._parse_flow_checklist_json(
        json.dumps(wrapped),
        request="update src/example.py",
    )
    assert checklist is not None
    assert len(checklist.steps) >= 3
    assert checklist.steps[0].title.startswith("Inspect src/example.py")


def test_preview_execution_checklist_uses_planner_and_persists_to_flow_memory(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    monkeypatch.setattr(
        agent,
        "_plan_checklist_with_source",
        lambda request, flow_context=None: (_fixed_checklist(), [], "planner"),
    )

    preview = agent.preview_execution_checklist("implement plan.")
    assert preview["prechecklist_source"] == "planner"
    assert isinstance(preview.get("prechecklist"), dict)
    assert preview["prechecklist"]["objective"] == "Implement request"
    flow_id = preview.get("flow_id")
    assert isinstance(flow_id, str) and flow_id
    summary = agent.flow_summary(flow_id)
    assert isinstance(summary, dict)
    checklist = summary.get("checklist")
    assert isinstance(checklist, dict)
    assert checklist.get("objective") == "Implement request"


def test_preview_execution_checklist_reports_repair_source(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    monkeypatch.setattr(
        agent,
        "_plan_checklist_with_source",
        lambda request, flow_context=None: (_fixed_checklist(), ["planner parse failed; attempting repair"], "planner_repair"),
    )
    preview = agent.preview_execution_checklist("implement plan.")
    assert preview["prechecklist_source"] == "planner_repair"
    assert str(preview.get("prechecklist_warning", "")).strip() == ""


def test_preview_execution_checklist_surfaces_deterministic_fallback_warning(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    monkeypatch.setattr(
        agent,
        "_plan_checklist_with_source",
        lambda request, flow_context=None: (_fixed_checklist(), ["planner parse failed"], "deterministic_fallback"),
    )
    preview = agent.preview_execution_checklist("implement plan.")
    assert preview["prechecklist_source"] == "deterministic_fallback"
    assert "deterministic fallback checklist" in str(preview.get("prechecklist_warning", "")).lower()
