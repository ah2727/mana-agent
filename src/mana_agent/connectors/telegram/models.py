from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class TelegramDocument(BaseModel):
    file_id: str
    file_unique_id: str = ""
    file_name: str = ""
    mime_type: str = ""
    file_size: int = 0


class TelegramReply(BaseModel):
    message_id: int
    sender_user_id: int | None = None
    text: str = ""
    sender_is_bot: bool = False


class TelegramUpdate(BaseModel):
    update_id: int
    message_id: int
    chat_id: int
    chat_type: str
    message_thread_id: int | None = None
    sender_user_id: int
    username: str = ""
    first_name: str = ""
    last_name: str = ""
    text: str = ""
    reply_to: TelegramReply | None = None
    document: TelegramDocument | None = None
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    transport: Literal["polling", "webhook"]
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)

    def conversation_key(self, bot_id: int) -> str:
        sender = self.sender_user_id if self.chat_type == "private" else 0
        topic = self.message_thread_id or 0
        return f"{int(bot_id)}:{self.chat_id}:{topic}:{sender}"


class TelegramBotIdentity(BaseModel):
    id: int
    username: str = ""
    first_name: str = ""


class TelegramJob(BaseModel):
    update_id: int
    conversation_key: str
    payload: dict[str, Any]
    status: Literal["queued", "processing", "completed", "failed"]
    attempts: int = 0
    available_at: float = 0
