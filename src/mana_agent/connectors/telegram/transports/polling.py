from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..errors import TelegramConflictError, TelegramRateLimitError
from ..normalizer import TelegramUpdateNormalizer
from ..observability import emit_telegram_event

logger = logging.getLogger(__name__)


class TelegramPollingTransport:
    def __init__(self, *, client: Any, store: Any, task_queue: Any, normalizer: TelegramUpdateNormalizer, bot_id: int, token: str, config: Any, sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        self.client = client
        self.store = store
        self.task_queue = task_queue
        self.normalizer = normalizer
        self.bot_id = int(bot_id)
        self.config = config
        self.sleeper = sleeper
        fingerprint = hashlib.sha256(token.encode("utf-8")).hexdigest()[:20]
        self.lock_path = Path(store.path).parent / f"poller-{fingerprint}.lock"
        self._stop = asyncio.Event()
        self._lock_handle: Any = None

    def _acquire_lock(self) -> None:
        self.lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        handle = self.lock_path.open("a+")
        try:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (ImportError, BlockingIOError, OSError) as exc:
            handle.close()
            raise TelegramConflictError(409, "Another polling worker is already using this Telegram bot.") from exc
        self._lock_handle = handle

    def _release_lock(self) -> None:
        if self._lock_handle is None:
            return
        try:
            import fcntl
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        self._lock_handle.close()
        self._lock_handle = None

    async def run(self) -> None:
        self._acquire_lock()
        backoff = 1.0
        try:
            await self.client.delete_webhook(drop_pending_updates=self.config.drop_pending_updates)
            emit_telegram_event("transport.started", transport="polling", bot_id=self.bot_id)
            while not self._stop.is_set():
                try:
                    offset = self.store.polling_offset()
                    payloads = await self.client.get_updates(
                        offset=offset, timeout=self.config.timeout_seconds,
                        allowed_updates=["message", "edited_message"],
                    )
                    for payload in payloads:
                        update = self.normalizer.normalize(payload, transport="polling")
                        inserted = self.store.persist(update, conversation_key=update.conversation_key(self.bot_id), commit_offset=update.update_id + 1)
                        emit_telegram_event("update.received", update_id=update.update_id, chat_id=update.chat_id, transport="polling", duplicate=not inserted)
                        if inserted:
                            self.task_queue.notify()
                    backoff = 1.0
                except TelegramRateLimitError as exc:
                    emit_telegram_event("rate_limited", transport="polling", retry_after=exc.retry_after)
                    await self.sleeper(float(exc.retry_after or backoff))
                except TelegramConflictError:
                    raise
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning("Telegram polling reconnect", exc_info=True)
                    await self.sleeper(backoff)
                    backoff = min(float(self.config.reconnect_max_seconds), backoff * 2)
        finally:
            emit_telegram_event("transport.stopped", transport="polling", bot_id=self.bot_id)
            self._release_lock()

    async def stop(self) -> None:
        self._stop.set()
