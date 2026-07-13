from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mana_agent.connectors.telegram.access import TelegramAccessController
from mana_agent.connectors.telegram.config import TelegramConfig
from mana_agent.connectors.telegram.models import TelegramUpdate
from mana_agent.connectors.telegram.normalizer import TelegramUpdateNormalizer
from mana_agent.connectors.telegram.renderer import TelegramMessageRenderer
from mana_agent.connectors.telegram.store import TelegramUpdateStore
from mana_agent.connectors.telegram.transports.webhook import TelegramWebhookReceiver, create_telegram_webhook_router


def payload(update_id: int = 1, *, chat_id: int = 20, chat_type: str = "private", thread_id: int | None = None) -> dict:
    message = {
        "message_id": update_id + 100,
        "date": 1_700_000_000,
        "chat": {"id": chat_id, "type": chat_type},
        "from": {"id": 10, "username": "user", "first_name": "U"},
        "text": "hello",
    }
    if thread_id is not None:
        message["message_thread_id"] = thread_id
    return {"update_id": update_id, "message": message}


def normalized(update_id: int = 1, *, chat_id: int = 20, chat_type: str = "private", thread_id: int | None = None) -> TelegramUpdate:
    return TelegramUpdateNormalizer().normalize(payload(update_id, chat_id=chat_id, chat_type=chat_type, thread_id=thread_id), transport="polling")


def test_config_transport_selection_and_runtime_validation(monkeypatch, tmp_path: Path) -> None:
    config = TelegramConfig(default_repository=str(tmp_path))
    assert config.transport == "polling"
    assert config.effective_transport == "polling"
    auto = TelegramConfig(transport="auto", default_repository=str(tmp_path))
    assert auto.effective_transport == "polling"
    webhook = TelegramConfig(transport="auto", webhook={"public_url": "https://bot.example"}, default_repository=str(tmp_path))
    assert webhook.effective_transport == "webhook"
    monkeypatch.setenv("TOKEN_TEST", "123:secret")
    enabled = TelegramConfig(enabled=True, bot_token_env="TOKEN_TEST", default_repository=str(tmp_path))
    enabled.validate_runtime()


def test_normalizer_preserves_topic_reply_and_document() -> None:
    value = payload(thread_id=77)
    value["message"]["reply_to_message"] = {"message_id": 7, "text": "prior", "from": {"id": 99, "is_bot": True}}
    value["message"]["document"] = {"file_id": "f", "file_unique_id": "u", "file_name": "a.txt", "mime_type": "text/plain", "file_size": 3}
    update = TelegramUpdateNormalizer().normalize(value, transport="webhook")
    assert update.message_thread_id == 77
    assert update.reply_to and update.reply_to.sender_is_bot
    assert update.document and update.document.file_name == "a.txt"
    assert update.transport == "webhook"


def test_access_is_denied_by_default_and_uses_numeric_ids(tmp_path: Path) -> None:
    update = normalized()
    denied = TelegramAccessController(TelegramConfig(default_repository=str(tmp_path)), bot_id=1)
    assert not denied.authorize(update).allowed
    allowed = TelegramAccessController(TelegramConfig(allowed_users=[10], default_repository=str(tmp_path)), bot_id=1)
    assert allowed.authorize(update).allowed


def test_group_activation_and_topic_session_isolation(tmp_path: Path) -> None:
    config = TelegramConfig(allowed_chats=[-100], groups_enabled=True, group_activation="mention", default_repository=str(tmp_path))
    controller = TelegramAccessController(config, bot_id=55, bot_username="mana_bot")
    quiet = normalized(chat_id=-100, chat_type="supergroup", thread_id=1)
    assert not controller.authorize(quiet).allowed
    quiet.text = "@mana_bot help"
    assert controller.authorize(quiet).allowed
    assert quiet.conversation_key(55) != normalized(chat_id=-100, chat_type="supergroup", thread_id=2).conversation_key(55)


def test_store_deduplicates_orders_lanes_and_recovers(tmp_path: Path) -> None:
    store = TelegramUpdateStore(tmp_path / "queue.sqlite")
    one, two, other = normalized(1), normalized(2), normalized(3, chat_id=30)
    assert store.persist(one, conversation_key="a", commit_offset=2)
    assert not store.persist(one, conversation_key="a", commit_offset=2)
    store.persist(two, conversation_key="a", commit_offset=3)
    store.persist(other, conversation_key="b", commit_offset=4)
    assert store.polling_offset() == 4
    first = store.claim(lease_seconds=10, now=100)
    second_lane = store.claim(lease_seconds=10, now=100)
    assert first and first.update_id == 1
    assert second_lane and second_lane.update_id == 3
    assert store.claim(lease_seconds=10, now=100) is None
    assert store.recover_abandoned(now=111) == 2
    retried = store.claim(lease_seconds=10, now=111)
    assert retried and retried.update_id == 1
    store.complete(1)
    next_job = store.claim(lease_seconds=10, now=111)
    assert next_job and next_job.update_id == 2
    store.append_history("session-a", "question", "answer")
    assert store.history("session-a") == [("question", "answer")]


def test_store_retries_then_dead_letters(tmp_path: Path) -> None:
    store = TelegramUpdateStore(tmp_path / "queue.sqlite")
    store.persist(normalized(), conversation_key="a")
    job = store.claim(lease_seconds=10)
    assert job
    assert store.fail(1, "temporary", max_attempts=2, retry_delay_seconds=1, transient=True) == "queued"
    job = store.claim(lease_seconds=10, now=10**12)
    assert job
    assert store.fail(1, "temporary", max_attempts=2, retry_delay_seconds=1, transient=True) == "failed"
    assert store.stats()["failed"] == 1


def test_renderer_escapes_and_splits_within_limit() -> None:
    renderer = TelegramMessageRenderer(parse_mode="MarkdownV2", max_length=256)
    chunks = renderer.render("*unsafe*\n" + ("word " * 200))
    assert "\\*unsafe\\*" in chunks[0]
    assert all(len(chunk) <= 256 for chunk in chunks)
    code_chunks = renderer.render("```python\n" + ("print('x')\n" * 80) + "```")
    assert all(len(chunk) <= 256 for chunk in code_chunks)
    assert all(chunk.count("```") % 2 == 0 for chunk in code_chunks)


class _Queue:
    def __init__(self) -> None:
        self.notifications = 0
    def notify(self) -> None:
        self.notifications += 1


def test_webhook_validates_secret_deduplicates_and_acknowledges(tmp_path: Path) -> None:
    store = TelegramUpdateStore(tmp_path / "queue.sqlite")
    queue = _Queue()
    receiver = TelegramWebhookReceiver(secret="x" * 32, store=store, task_queue=queue, normalizer=TelegramUpdateNormalizer(), bot_id=5, max_request_bytes=10000)
    app = FastAPI()
    app.include_router(create_telegram_webhook_router(receiver, path="/hook"))
    client = TestClient(app)
    assert client.post("/hook", json=payload()).status_code == 403
    headers = {"X-Telegram-Bot-Api-Secret-Token": "x" * 32}
    assert client.post("/hook", json=payload(), headers=headers).status_code == 200
    assert client.post("/hook", json=payload(), headers=headers).status_code == 200
    assert store.stats()["queued"] == 1
    assert queue.notifications == 1
    assert client.post("/hook", content=b"bad", headers={**headers, "content-type": "application/json"}).status_code == 400


class _Client:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []
        self.actions = 0
    async def send_chat_action(self, *_args, **_kwargs):
        self.actions += 1
    async def send_message(self, chat_id, text, **_kwargs):
        self.sent.append((chat_id, text))
        return {}


class _Gateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
    def create_session(self) -> str:
        return f"session-{len(self.calls)}"
    async def send(self, session_id: str, text: str) -> str:
        self.calls.append((session_id, text))
        return "answer"
    async def status(self, _session_id: str) -> str:
        return "ready"
    async def cancel(self, _session_id: str) -> bool:
        return False


def test_pipeline_reuses_chat_gateway_and_isolates_conversations(tmp_path: Path) -> None:
    from mana_agent.connectors.telegram.chat import TelegramConversationRouter
    from mana_agent.connectors.telegram.pipeline import TelegramUpdateProcessor
    store = TelegramUpdateStore(tmp_path / "queue.sqlite")
    gateway, client = _Gateway(), _Client()
    router = TelegramConversationRouter(store, gateway)
    access = TelegramAccessController(TelegramConfig(open_access=True, default_repository=str(tmp_path)), bot_id=5)
    processor = TelegramUpdateProcessor(client=client, access=access, router=router, gateway=gateway, renderer=TelegramMessageRenderer(parse_mode="plain"))
    async def run() -> None:
        await processor.process(normalized(1, chat_id=20), conversation_key="a")
        await processor.process(normalized(2, chat_id=20), conversation_key="a")
        await processor.process(normalized(3, chat_id=30), conversation_key="b")
    asyncio.run(run())
    assert gateway.calls[0][0] == gateway.calls[1][0]
    assert gateway.calls[2][0] != gateway.calls[0][0]
    assert [text for _, text in client.sent] == ["answer", "answer", "answer"]
