from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("mana_agent.telegram.events")
_ALLOWED_FIELDS = {
    "update_id", "chat_id", "transport", "duplicate", "status", "attempts",
    "duration_ms", "retry_after", "queue_depth", "effective_transport", "bot_id",
}


def emit_telegram_event(event: str, **fields: Any) -> None:
    """Emit a structured event without message content, secrets, or tool output."""
    safe = {key: value for key, value in fields.items() if key in _ALLOWED_FIELDS}
    logger.info("telegram.%s", event, extra={"telegram_event": event, **safe})
