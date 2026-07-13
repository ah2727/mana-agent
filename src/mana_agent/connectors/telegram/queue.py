from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .errors import TelegramApiError
from .models import TelegramUpdate
from .observability import emit_telegram_event

logger = logging.getLogger(__name__)


class TelegramTaskQueue:
    def __init__(self, store: Any, processor: Any, *, concurrency: int, lease_seconds: int, max_attempts: int, retry_delay_seconds: int) -> None:
        self.store = store
        self.processor = processor
        self.concurrency = concurrency
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self._stop = asyncio.Event()
        self._wake = asyncio.Event()
        self._workers: list[asyncio.Task[None]] = []
        self._active: set[int] = set()

    async def start(self) -> None:
        self.store.recover_abandoned()
        self._stop.clear()
        self._workers = [asyncio.create_task(self._worker(), name=f"telegram-worker-{index}") for index in range(self.concurrency)]

    def notify(self) -> None:
        self._wake.set()

    async def _worker(self) -> None:
        while not self._stop.is_set():
            job = self.store.claim(lease_seconds=self.lease_seconds)
            if job is None:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
                continue
            self._active.add(job.update_id)
            started = time.monotonic()
            try:
                update = TelegramUpdate.model_validate(job.payload)
                await self.processor.process(update, conversation_key=job.conversation_key)
            except asyncio.CancelledError:
                self.store.requeue(job.update_id)
                raise
            except Exception as exc:
                transient = (isinstance(exc, TelegramApiError) and exc.transient) or isinstance(exc, (ConnectionError, OSError, TimeoutError))
                delay = max(self.retry_delay_seconds, int(exc.retry_after or 0)) if isinstance(exc, TelegramApiError) else self.retry_delay_seconds
                failure_status = self.store.fail(job.update_id, str(exc), max_attempts=self.max_attempts, retry_delay_seconds=delay, transient=transient)
                logger.warning("Telegram update processing failed", extra={"update_id": job.update_id, "transient": transient})
                emit_telegram_event("job.failed", update_id=job.update_id, attempts=job.attempts, status=failure_status, duration_ms=round((time.monotonic() - started) * 1000, 2))
            else:
                self.store.complete(job.update_id)
                emit_telegram_event("job.completed", update_id=job.update_id, attempts=job.attempts, status="completed", duration_ms=round((time.monotonic() - started) * 1000, 2))
            finally:
                self._active.discard(job.update_id)

    async def stop(self, *, drain_seconds: float = 10.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._workers:
            done, pending = await asyncio.wait(self._workers, timeout=drain_seconds)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        self._workers.clear()
