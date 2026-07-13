from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from .errors import TelegramError
from .models import TelegramDocument, TelegramReply, TelegramUpdate


class TelegramUpdateNormalizer:
    def normalize(self, payload: dict[str, Any], *, transport: Literal["polling", "webhook"]) -> TelegramUpdate:
        try:
            message = payload.get("message") or payload.get("edited_message")
            if not isinstance(message, dict):
                raise KeyError("message")
            chat = message["chat"]
            sender = message["from"]
            reply_payload = message.get("reply_to_message")
            reply = None
            if isinstance(reply_payload, dict):
                reply_sender = reply_payload.get("from") or {}
                reply = TelegramReply(
                    message_id=int(reply_payload["message_id"]),
                    sender_user_id=int(reply_sender["id"]) if reply_sender.get("id") is not None else None,
                    text=str(reply_payload.get("text") or reply_payload.get("caption") or ""),
                    sender_is_bot=bool(reply_sender.get("is_bot")),
                )
            document_payload = message.get("document")
            document = TelegramDocument.model_validate(document_payload) if isinstance(document_payload, dict) else None
            timestamp = message.get("date")
            received = datetime.fromtimestamp(int(timestamp), timezone.utc) if timestamp else datetime.now(timezone.utc)
            return TelegramUpdate(
                update_id=int(payload["update_id"]), message_id=int(message["message_id"]),
                chat_id=int(chat["id"]), chat_type=str(chat.get("type") or "unknown"),
                message_thread_id=int(message["message_thread_id"]) if message.get("message_thread_id") is not None else None,
                sender_user_id=int(sender["id"]), username=str(sender.get("username") or ""),
                first_name=str(sender.get("first_name") or ""), last_name=str(sender.get("last_name") or ""),
                text=str(message.get("text") or message.get("caption") or ""), reply_to=reply,
                document=document, received_at=received, transport=transport, raw=payload,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TelegramError("Invalid Telegram update payload.") from exc
