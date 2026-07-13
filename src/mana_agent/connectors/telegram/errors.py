from __future__ import annotations


class TelegramError(RuntimeError):
    """Base error whose message is safe to show to an administrator."""


class TelegramConfigurationError(TelegramError):
    pass


class TelegramApiError(TelegramError):
    def __init__(self, status_code: int, description: str, *, retry_after: int | None = None) -> None:
        self.status_code = int(status_code)
        self.description = str(description or "Telegram API request failed.")
        self.retry_after = retry_after
        super().__init__(f"Telegram API error {self.status_code}: {self.description}")

    @property
    def transient(self) -> bool:
        return self.status_code == 429 or self.status_code >= 500


class TelegramConflictError(TelegramApiError):
    pass


class TelegramRateLimitError(TelegramApiError):
    pass


class TelegramQueueError(TelegramError):
    pass
