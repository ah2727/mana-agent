from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from mana_agent.connectors.telegram.client import TelegramBotClient
from mana_agent.connectors.telegram.config import TelegramPollingConfig
from mana_agent.connectors.telegram.errors import TelegramApiError, TelegramConflictError, TelegramRateLimitError
from mana_agent.connectors.telegram.normalizer import TelegramUpdateNormalizer
from mana_agent.connectors.telegram.store import TelegramUpdateStore
from mana_agent.connectors.telegram.transports.polling import TelegramPollingTransport


class _Response:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self.payload, self.status = payload, status
    def __enter__(self): return self
    def __exit__(self, *_args): return None
    def read(self, *_args): return json.dumps(self.payload).encode()


def test_client_methods_use_single_bot_api_boundary() -> None:
    requests = []
    def opener(request, timeout):
        requests.append((request.full_url, json.loads(request.data), timeout))
        method = request.full_url.rsplit("/", 1)[-1]
        result = {"id": 5, "username": "mana", "first_name": "Mana"} if method == "getMe" else True
        return _Response({"ok": True, "result": result})
    client = TelegramBotClient("token", opener=opener)
    async def run():
        identity = await client.get_me()
        assert identity.id == 5
        await client.delete_webhook(drop_pending_updates=True)
        await client.set_webhook("https://example/hook", "s" * 32)
        await client.close()
    asyncio.run(run())
    assert requests[0][0].endswith("/getMe")
    assert requests[1][1]["drop_pending_updates"] is True
    assert "token" not in repr(client.__dict__.keys())


@pytest.mark.parametrize(
    ("code", "error_type"),
    [(401, TelegramApiError), (403, TelegramApiError), (409, TelegramConflictError), (429, TelegramRateLimitError)],
)
def test_client_maps_api_errors_and_preserves_description(code: int, error_type: type[Exception]) -> None:
    def opener(_request, timeout):
        return _Response({"ok": False, "error_code": code, "description": "specific safe diagnostic", "parameters": {"retry_after": 7}})
    client = TelegramBotClient("secret-token", opener=opener)
    with pytest.raises(error_type) as caught:
        asyncio.run(client.get_me())
    assert "specific safe diagnostic" in str(caught.value)
    assert "secret-token" not in str(caught.value)
    if code == 429:
        assert caught.value.retry_after == 7


class _Queue:
    def __init__(self) -> None: self.notifications = 0
    def notify(self): self.notifications += 1


class _PollingClient:
    def __init__(self) -> None:
        self.deleted = []
        self.calls = 0
        self.transport = None
    async def delete_webhook(self, **kwargs):
        self.deleted.append(kwargs)
        return True
    async def get_updates(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            await self.transport.stop()
            return [{"update_id": 8, "message": {"message_id": 1, "date": 1, "chat": {"id": 2, "type": "private"}, "from": {"id": 3}, "text": "x"}}]
        return []


def test_polling_deletes_webhook_persists_before_offset_and_notifies(tmp_path: Path) -> None:
    client, queue = _PollingClient(), _Queue()
    store = TelegramUpdateStore(tmp_path / "q.sqlite")
    transport = TelegramPollingTransport(
        client=client, store=store, task_queue=queue, normalizer=TelegramUpdateNormalizer(),
        bot_id=4, token="token", config=TelegramPollingConfig(drop_pending_updates=True),
    )
    client.transport = transport
    asyncio.run(transport.run())
    assert client.deleted == [{"drop_pending_updates": True}]
    assert store.polling_offset() == 9
    assert store.stats()["queued"] == 1
    assert queue.notifications == 1


def test_polling_lock_prevents_two_workers(tmp_path: Path) -> None:
    client, queue = _PollingClient(), _Queue()
    store = TelegramUpdateStore(tmp_path / "q.sqlite")
    one = TelegramPollingTransport(client=client, store=store, task_queue=queue, normalizer=TelegramUpdateNormalizer(), bot_id=1, token="same", config=TelegramPollingConfig())
    two = TelegramPollingTransport(client=client, store=store, task_queue=queue, normalizer=TelegramUpdateNormalizer(), bot_id=1, token="same", config=TelegramPollingConfig())
    one._acquire_lock()
    try:
        with pytest.raises(TelegramConflictError):
            two._acquire_lock()
    finally:
        one._release_lock()
