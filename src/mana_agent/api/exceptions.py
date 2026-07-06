from __future__ import annotations


class ManaApiError(Exception):
    """API error with a stable JSON response shape."""

    def __init__(self, status_code: int, detail: str, *, error: str | None = None) -> None:
        self.status_code = status_code
        self.detail = detail
        self.error = error

