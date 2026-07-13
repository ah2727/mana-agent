from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from .models import TelegramUpdate


@dataclass(slots=True)
class CommandContext:
    update: TelegramUpdate
    conversation_key: str
    session_id: str
    router: object
    gateway: object


CommandHandler = Callable[[CommandContext], Awaitable[str]]


@dataclass(frozen=True, slots=True)
class CommandDefinition:
    name: str
    description: str
    handler: CommandHandler


class TelegramCommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, CommandDefinition] = {}
        for definition in (
            CommandDefinition("start", "Connect this conversation to Mana-Agent", self._start),
            CommandDefinition("help", "Show available commands", self._help),
            CommandDefinition("status", "Show this conversation's task status", self._status),
            CommandDefinition("new", "Start a new conversation", self._new),
            CommandDefinition("cancel", "Request cancellation of the active task", self._cancel),
            CommandDefinition("id", "Show Telegram user, chat, and topic IDs", self._id),
        ):
            self._commands[definition.name] = definition

    @property
    def definitions(self) -> tuple[CommandDefinition, ...]:
        return tuple(self._commands.values())

    def parse(self, text: str) -> str | None:
        first = str(text or "").strip().split(maxsplit=1)[0]
        if not first.startswith("/"):
            return None
        return first[1:].split("@", 1)[0].casefold()

    async def dispatch(self, name: str, context: CommandContext) -> str:
        definition = self._commands.get(name)
        if definition is None:
            return "Unknown command. Use /help to list available commands."
        return await definition.handler(context)

    async def _start(self, _context: CommandContext) -> str:
        return "This bot connects this Telegram conversation to Mana-Agent. Send a task or use /help."

    async def _help(self, _context: CommandContext) -> str:
        return "Available commands:\n" + "\n".join(f"/{item.name} — {item.description}" for item in self.definitions)

    async def _status(self, context: CommandContext) -> str:
        value = await context.gateway.status(context.session_id)
        return f"Mana-Agent session status: {value}."

    async def _new(self, context: CommandContext) -> str:
        context.router.reset(context.conversation_key)
        return "Started a new Mana-Agent conversation for this Telegram context."

    async def _cancel(self, context: CommandContext) -> str:
        cancelled = await context.gateway.cancel(context.session_id)
        return "Cancellation requested." if cancelled else "The active runtime does not support cooperative cancellation."

    async def _id(self, context: CommandContext) -> str:
        topic = context.update.message_thread_id if context.update.message_thread_id is not None else "none"
        return f"User ID: {context.update.sender_user_id}\nChat ID: {context.update.chat_id}\nTopic ID: {topic}"
