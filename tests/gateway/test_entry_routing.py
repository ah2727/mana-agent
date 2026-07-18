from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mana_agent.gateway import (
    AgentChatGateway,
    EntryRouteRegistry,
    EntryRouter,
    RouteAvailability,
    RouteRegistration,
)
from mana_agent.gateway.entry_routing import gmail_route_availability
from mana_agent.workspaces.service import WorkspaceService


class _RouteModel:
    def __init__(self, *routes: str) -> None:
        self.routes = list(routes)
        self.payloads: list[dict[str, Any]] = []

    def invoke(self, messages: list[Any]) -> Any:
        self.payloads.append(json.loads(messages[-1].content))
        route = self.routes.pop(0) if self.routes else "conversation"
        return SimpleNamespace(
            content=json.dumps(
                {
                    "route": route,
                    "confidence": 0.98,
                    "reason": f"selected {route}",
                    "reuse_active_route": len(self.payloads) > 1,
                }
            )
        )


class _AskAgent:
    def __init__(self, response: Any | None = None) -> None:
        self.response = response or SimpleNamespace(
            answer="Latest Gmail: Subject: hello",
            sources=[],
            warnings=[],
            trace=[{"tool_name": "email_search", "status": "ok"}],
        )
        self.calls: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


class _AskService:
    def __init__(self, ask_agent: _AskAgent | None = None) -> None:
        self.ask_agent = ask_agent or _AskAgent()
        self.qna_chain = SimpleNamespace(llm=None, chat=lambda question: "chat")
        self.entry_router = SimpleNamespace(llm=None)


class _ChatService:
    def __init__(self, ask_service: _AskService) -> None:
        self._ask_service = ask_service
        self.conversation_calls: list[str] = []

    def ask_conversation(self, question: str) -> str:
        self.conversation_calls.append(question)
        return "ordinary conversation"

    def ask(self, question: str, **kwargs: Any) -> Any:
        return SimpleNamespace(answer="repository answer", sources=[], warnings=[], trace=[])


class _CodingAgent:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.session_id = "bootstrap-session"

    def generate(self, request: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(request)
        return {"answer": "coding route", "changed_files": [], "warnings": []}

    generate_auto_execute = generate
    generate_dir_mode = generate

    def get_active_flow_id(self) -> None:
        return None

    def flow_summary(self, flow_id: str) -> None:
        return None

    def reset_flow(self, flow_id: str) -> str:
        return flow_id

    def _tool_policy_for_request(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"allowed_tools": ["read_file"]}


def _registry(gmail: RouteAvailability | None = None) -> EntryRouteRegistry:
    registry = EntryRouteRegistry()
    for name, description in (
        ("conversation", "ordinary conversation"),
        ("coding", "Codex coding"),
        ("gmail", "Gmail inbox"),
        ("calendar", "calendar"),
        ("search", "public search"),
        ("repository", "repository inspection"),
        ("automation", "automation"),
        ("unsupported", "safe stop"),
    ):
        availability = gmail if name == "gmail" and gmail is not None else RouteAvailability(True)
        registry.register(
            RouteRegistration(
                name,  # type: ignore[arg-type]
                description,
                lambda value=availability: value,
            )
        )
    return registry


def _gateway(
    root: Path,
    model: _RouteModel,
    *,
    gmail: RouteAvailability | None = None,
    ask_agent: _AskAgent | None = None,
    coding_agent: _CodingAgent | None = None,
) -> tuple[AgentChatGateway, _ChatService, _AskAgent]:
    agent = ask_agent or _AskAgent()
    ask_service = _AskService(agent)
    chat_service = _ChatService(ask_service)
    registry = _registry(gmail)
    gateway = AgentChatGateway(
        root,
        coding_agent=coding_agent is not None,
        coding_agent_instance=coding_agent,
        agent_tools=True,
        chat_service=chat_service,
        entry_route_registry=registry,
        entry_router=EntryRouter(llm=model, registry=registry),
    )
    return gateway, chat_service, agent


def test_latest_gmail_routes_to_connector_and_preserves_identifiers(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    gateway, chat, ask_agent = _gateway(tmp_path, _RouteModel("gmail"))
    session_id = gateway.create_session(frontend="test")
    result = gateway.process_turn(session_id, "Check my latest Gmail", turn_id="turn_exact")

    assert result.mode == "route-gmail"
    assert not chat.conversation_calls
    assert ask_agent.calls[0]["flow_id"] == session_id
    assert ask_agent.calls[0]["run_id"] == "turn_exact"
    assert result.payload["session_id"] == session_id
    assert result.payload["conversation_id"] == session_id
    assert result.payload["turn_id"] == "turn_exact"


def test_missing_gmail_configuration_returns_truthful_setup_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    unavailable = RouteAvailability(
        False,
        configured=False,
        authorized=False,
        reason="No enabled Gmail account is configured.",
        setup_action="Run `mana-agent connector email add --provider gmail ...`.",
    )
    gateway, chat, ask_agent = _gateway(tmp_path, _RouteModel("gmail"), gmail=unavailable)
    result = gateway.process_turn(gateway.create_session(frontend="test"), "Check Gmail")

    assert result.mode == "route-gmail-unavailable"
    assert "connector email add" in result.answer
    assert not chat.conversation_calls
    assert not ask_agent.calls


def test_gmail_availability_reads_live_account_and_credential_registry(monkeypatch) -> None:
    from mana_agent.connectors.email.models import (
        EmailAccount,
        EmailAddress,
        EmailPermission,
    )

    account = EmailAccount(
        id="gmail-live",
        provider="gmail",
        address=EmailAddress(address="me@example.com"),
        granted_permissions={EmailPermission.READ},
        secret_ref="credential-ref",
    )
    monkeypatch.setattr(
        "mana_agent.connectors.email.config.load_accounts",
        lambda: [account],
    )
    monkeypatch.setattr(
        "mana_agent.connectors.email.auth.credential_store.CredentialStore.get",
        lambda self, reference: {"token": "present"},
    )

    availability = gmail_route_availability()

    assert availability.available is True
    assert availability.configured is True
    assert availability.authorized is True
    assert availability.details["account_id"] == "gmail-live"


def test_gmail_provider_authorization_details_are_not_replaced(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    provider_error = (
        "email_authorization_failed provider=gmail provider_status=403 "
        "reconnect_required=true missing_scope=email.read"
    )
    ask_agent = _AskAgent(
        SimpleNamespace(answer=provider_error, sources=[], warnings=[], trace=[])
    )
    gateway, chat, _ = _gateway(tmp_path, _RouteModel("gmail"), ask_agent=ask_agent)
    result = gateway.process_turn(gateway.create_session(frontend="test"), "Read latest Gmail")

    assert result.answer == provider_error
    assert result.mode == "route-gmail"
    assert not chat.conversation_calls


def test_conversation_and_coding_use_their_selected_routes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    model = _RouteModel("conversation", "coding")
    coding = _CodingAgent()
    gateway, chat, _ = _gateway(tmp_path, model, coding_agent=coding)
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.handle_small_direct_edit",
        lambda *args, **kwargs: SimpleNamespace(handled=False),
    )
    session_id = gateway.create_session(frontend="test")

    conversation = gateway.process_turn(session_id, "Hello, how are you?")
    coding_result = gateway.process_turn(session_id, "Change the parser implementation")

    assert conversation.mode == "route-conversation"
    assert conversation.payload["lane_id"] == "research"
    assert len(chat.conversation_calls) == 1
    assert coding_result.used_coding_agent is True
    assert coding_result.payload["lane_id"] == "coding"
    assert coding.calls
    assert coding.session_id == session_id
    assert coding_result.payload["session_id"] == session_id


def test_followup_gmail_reuses_one_session_and_supplies_previous_route(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    model = _RouteModel("gmail", "gmail")
    gateway, _, ask_agent = _gateway(tmp_path, model)
    session_id = gateway.create_session(frontend="test")

    gateway.process_turn(session_id, "Check latest Gmail")
    gateway.process_turn(session_id, "Open the first one")

    assert len(ask_agent.calls) == 2
    assert {call["flow_id"] for call in ask_agent.calls} == {session_id}
    assert model.payloads[1]["context"]["previous_route"] == "gmail"
    assert {row["session_id"] for row in gateway.session_messages(session_id)} == {session_id}
    sessions = WorkspaceService().store.list_sessions()
    assert [item.session_id for item in sessions if item.status == "active"] == [session_id]


def test_invalid_entry_decision_stops_without_connector_refusal(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    model = _RouteModel("not-a-route")
    gateway, chat, ask_agent = _gateway(tmp_path, model)
    result = gateway.process_turn(gateway.create_session(frontend="test"), "Check Gmail")

    assert result.mode == "route-error"
    assert result.payload["route"] == "unsupported"
    assert "integration" not in result.answer.lower()
    assert not chat.conversation_calls
    assert not ask_agent.calls


def test_session_close_new_history_and_stale_finalization(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    gateway, _, _ = _gateway(tmp_path, _RouteModel("conversation"))
    first = gateway.create_session(frontend="test")
    gateway.process_turn(first, "Remember this")
    assert gateway.close_session(first) == first
    assert gateway.close_session(first) == first

    service = WorkspaceService()
    first_record = service.store.get_session(first)
    assert first_record.status == "closed"
    assert first_record.opened_at
    assert first_record.closed_at
    assert gateway.session_messages(first)

    second_gateway, _, _ = _gateway(tmp_path, _RouteModel("conversation"))
    second = second_gateway.create_session(frontend="test")
    assert second != first

    stale = service.create_session(tmp_path)
    stale.owner_pid = 999_999_999
    service.store.save_session(stale)
    finalized = service.finalize_stale_sessions(tmp_path)
    assert stale.session_id in {item.session_id for item in finalized}
    assert service.store.get_session(stale.session_id).status == "abandoned"


def test_new_closes_previous_and_opens_fresh_session(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    gateway, _, _ = _gateway(tmp_path, _RouteModel("conversation"))
    first = gateway.create_session(frontend="test")
    second = gateway.start_new_conversation(first, frontend="test")

    service = WorkspaceService()
    assert second != first
    assert service.store.get_session(first).status == "closed"
    assert service.store.get_session(second).status == "active"
