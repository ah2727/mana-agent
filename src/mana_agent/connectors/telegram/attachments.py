from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from mana_agent.documents.service import DocumentService
from mana_agent.workspaces.paths import mana_home

from .config import TelegramAttachmentConfig
from .models import TelegramDocument

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def safe_attachment_filename(value: str) -> str:
    name = Path(str(value or "attachment").replace("\\", "/")).name
    cleaned = _SAFE_NAME.sub("_", name).strip("._")
    return cleaned[:180] or "attachment"


class TelegramAttachmentManager:
    def __init__(self, client: Any, config: TelegramAttachmentConfig) -> None:
        self.client = client
        self.config = config
        self.root = mana_home() / "connectors" / "telegram" / "attachments"

    async def prepare(self, document: TelegramDocument, *, session_id: str, update_id: int) -> tuple[str, Path]:
        if not self.config.enabled:
            raise ValueError("Telegram document attachments are disabled.")
        if document.file_size and document.file_size > self.config.max_bytes:
            raise ValueError("Telegram document exceeds the configured size limit.")
        if document.mime_type not in self.config.allowed_mime_types:
            raise ValueError("Telegram document type is not supported.")
        metadata = await self.client.get_file(document.file_id)
        file_path = str(metadata.get("file_path") or "")
        if not file_path:
            raise ValueError("Telegram did not return an attachment file path.")
        data = await self.client.download_file(file_path, max_bytes=self.config.max_bytes)
        directory = (self.root / safe_attachment_filename(session_id) / str(update_id)).resolve()
        if self.root.resolve() not in directory.parents:
            raise ValueError("Invalid Telegram attachment path.")
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        local = directory / safe_attachment_filename(document.file_name or Path(file_path).name)
        local.write_bytes(data)
        try:
            local.chmod(0o600)
        except OSError:
            pass
        parsed = DocumentService(directory).read(local.name, max_chunks=100)
        if not parsed.get("ok"):
            raise ValueError("Telegram document could not be read by Mana-Agent's document service.")
        content = "\n\n".join(str(item.get("content") or "") for item in parsed.get("chunks", []) if item.get("content"))
        return content[:100_000], directory

    @staticmethod
    def cleanup(directory: Path) -> None:
        shutil.rmtree(directory, ignore_errors=True)
