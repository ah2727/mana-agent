from __future__ import annotations

import hmac
from typing import Any

from fastapi import APIRouter, Request, Response

from ..errors import TelegramError, TelegramQueueError
from ..normalizer import TelegramUpdateNormalizer
from ..observability import emit_telegram_event


class TelegramWebhookReceiver:
    def __init__(self, *, secret: str, store: Any, task_queue: Any, normalizer: TelegramUpdateNormalizer, bot_id: int, max_request_bytes: int) -> None:
        self.secret = secret
        self.store = store
        self.task_queue = task_queue
        self.normalizer = normalizer
        self.bot_id = int(bot_id)
        self.max_request_bytes = int(max_request_bytes)

    async def receive(self, request: Request) -> Response:
        supplied = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if not supplied or not hmac.compare_digest(supplied, self.secret):
            emit_telegram_event("webhook.validation_failed", transport="webhook")
            return Response(status_code=403)
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            return Response(status_code=415)
        declared = request.headers.get("content-length")
        if declared:
            try:
                if int(declared) > self.max_request_bytes:
                    return Response(status_code=413)
            except ValueError:
                return Response(status_code=400)
        body = await request.body()
        if len(body) > self.max_request_bytes:
            return Response(status_code=413)
        try:
            import json
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError
            update = self.normalizer.normalize(payload, transport="webhook")
            inserted = self.store.persist(update, conversation_key=update.conversation_key(self.bot_id))
            emit_telegram_event("update.received", update_id=update.update_id, chat_id=update.chat_id, transport="webhook", duplicate=not inserted)
        except TelegramQueueError:
            return Response(status_code=503)
        except (ValueError, UnicodeDecodeError, TelegramError):
            return Response(status_code=400)
        if inserted:
            self.task_queue.notify()
        return Response(status_code=200)


def create_telegram_webhook_router(receiver: TelegramWebhookReceiver, *, path: str) -> APIRouter:
    router = APIRouter(tags=["telegram"])

    @router.post(path, include_in_schema=False)
    async def telegram_webhook(request: Request) -> Response:
        return await receiver.receive(request)

    return router
