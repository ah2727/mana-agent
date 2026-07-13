from __future__ import annotations

import asyncio
import json
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .errors import TelegramApiError, TelegramConflictError, TelegramRateLimitError
from .models import TelegramBotIdentity


class TelegramBotClient:
    """Small Bot API client; tokens are never included in errors or logs."""

    def __init__(self, token: str, *, timeout_seconds: int = 30, opener: Callable[..., Any] = urlopen) -> None:
        if not token.strip():
            raise ValueError("Telegram bot token is required.")
        self._base_url = f"https://api.telegram.org/bot{token.strip()}"
        self._file_url = f"https://api.telegram.org/file/bot{token.strip()}"
        self._timeout = timeout_seconds
        self._opener = opener
        self._closed = False

    def _request_sync(self, method: str, payload: dict[str, Any] | None = None) -> Any:
        data = json.dumps(payload or {}).encode("utf-8")
        request = Request(f"{self._base_url}/{method}", data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with self._opener(request, timeout=self._timeout) as response:
                body = response.read()
                status = int(getattr(response, "status", 200))
        except HTTPError as exc:
            body = exc.read()
            status = int(exc.code)
        except (URLError, TimeoutError, OSError) as exc:
            raise TelegramApiError(503, "Telegram network request failed.") from exc
        try:
            result = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TelegramApiError(status, "Telegram returned an invalid response.") from exc
        if status >= 400 or not result.get("ok"):
            parameters = result.get("parameters") if isinstance(result, dict) else {}
            retry_after = int(parameters.get("retry_after")) if isinstance(parameters, dict) and parameters.get("retry_after") else None
            description = str(result.get("description") or "Telegram API request failed.")
            error_code = int(result.get("error_code") or status or 500)
            cls = TelegramRateLimitError if error_code == 429 else TelegramConflictError if error_code == 409 else TelegramApiError
            raise cls(error_code, description, retry_after=retry_after)
        return result.get("result")

    async def request(self, method: str, payload: dict[str, Any] | None = None) -> Any:
        if self._closed:
            raise TelegramApiError(503, "Telegram client is closed.")
        return await asyncio.to_thread(self._request_sync, method, payload)

    async def get_me(self) -> TelegramBotIdentity:
        return TelegramBotIdentity.model_validate(await self.request("getMe"))

    async def get_updates(self, *, offset: int | None, timeout: int, allowed_updates: list[str] | None = None) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"timeout": timeout}
        if offset is not None:
            payload["offset"] = offset
        if allowed_updates is not None:
            payload["allowed_updates"] = allowed_updates
        result = await self.request("getUpdates", payload)
        return list(result or [])

    async def send_message(self, chat_id: int, text: str, *, parse_mode: str | None = None, message_thread_id: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if message_thread_id is not None:
            payload["message_thread_id"] = message_thread_id
        return dict(await self.request("sendMessage", payload))

    async def edit_message_text(self, chat_id: int, message_id: int, text: str, *, parse_mode: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"chat_id": chat_id, "message_id": message_id, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        return dict(await self.request("editMessageText", payload))

    async def send_chat_action(self, chat_id: int, action: str = "typing", *, message_thread_id: int | None = None) -> bool:
        payload: dict[str, Any] = {"chat_id": chat_id, "action": action}
        if message_thread_id is not None:
            payload["message_thread_id"] = message_thread_id
        return bool(await self.request("sendChatAction", payload))

    async def get_file(self, file_id: str) -> dict[str, Any]:
        return dict(await self.request("getFile", {"file_id": file_id}))

    async def download_file(self, file_path: str, *, max_bytes: int) -> bytes:
        def download() -> bytes:
            request = Request(f"{self._file_url}/{file_path.lstrip('/')}", method="GET")
            with self._opener(request, timeout=self._timeout) as response:
                data = response.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise TelegramApiError(413, "Telegram attachment exceeds the configured size limit.")
            return data
        return await asyncio.to_thread(download)

    async def set_webhook(self, url: str, secret_token: str, *, drop_pending_updates: bool = False) -> bool:
        return bool(await self.request("setWebhook", {"url": url, "secret_token": secret_token, "drop_pending_updates": drop_pending_updates}))

    async def delete_webhook(self, *, drop_pending_updates: bool = False) -> bool:
        return bool(await self.request("deleteWebhook", {"drop_pending_updates": drop_pending_updates}))

    async def get_webhook_info(self) -> dict[str, Any]:
        return dict(await self.request("getWebhookInfo"))

    async def close(self) -> None:
        self._closed = True
