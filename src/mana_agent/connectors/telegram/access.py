from __future__ import annotations

from dataclasses import dataclass

from .config import TelegramConfig
from .models import TelegramUpdate


@dataclass(frozen=True, slots=True)
class AccessDecision:
    allowed: bool
    reason: str
    respond: bool = True


class TelegramAccessController:
    def __init__(self, config: TelegramConfig, *, bot_id: int, bot_username: str = "") -> None:
        self.config = config
        self.bot_id = int(bot_id)
        self.bot_username = bot_username.lstrip("@").casefold()

    def authorize(self, update: TelegramUpdate) -> AccessDecision:
        is_private = update.chat_type == "private"
        authorized = self.config.open_access or update.sender_user_id in self.config.allowed_users or update.chat_id in self.config.allowed_chats
        if not authorized:
            return AccessDecision(False, "not_allowed")
        if self.config.private_only and not is_private:
            return AccessDecision(False, "private_only", respond=False)
        if not is_private and not self.config.groups_enabled:
            return AccessDecision(False, "groups_disabled", respond=False)
        if not is_private and not self._group_active(update):
            return AccessDecision(False, "group_not_activated", respond=False)
        return AccessDecision(True, "allowed")

    def _group_active(self, update: TelegramUpdate) -> bool:
        if update.chat_id in self.config.always_active_chats or self.config.group_activation == "always":
            return True
        text = update.text.strip()
        if text.startswith("/"):
            return True
        if self.config.group_activation == "command":
            return False
        if update.reply_to and update.reply_to.sender_user_id == self.bot_id:
            return True
        if self.config.group_activation == "reply":
            return False
        return bool(self.bot_username and f"@{self.bot_username}" in text.casefold())
