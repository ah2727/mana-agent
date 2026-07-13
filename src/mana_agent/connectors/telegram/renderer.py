from __future__ import annotations

import html
import re

_MARKDOWN_V2 = re.compile(r"([_\*\[\]()~`>#+\-=|{}.!\\])")


class TelegramMessageRenderer:
    def __init__(self, *, parse_mode: str = "MarkdownV2", max_length: int = 4096) -> None:
        self.parse_mode = parse_mode
        self.max_length = min(4096, max(256, int(max_length)))

    def escape(self, text: str) -> str:
        if self.parse_mode == "plain":
            return text
        if self.parse_mode == "HTML":
            return html.escape(text, quote=False)
        parts = text.split("```")
        rendered: list[str] = []
        for index, part in enumerate(parts):
            if index % 2:
                rendered.append("```" + part.replace("\\", "\\\\").replace("`", "\\`") + "```")
            else:
                rendered.append(_MARKDOWN_V2.sub(r"\\\1", part))
        return "".join(rendered)

    def render(self, text: str, *, formatted: bool = True) -> list[str]:
        value = self.escape(str(text or "")) if formatted else str(text or "")
        return self._split(value)

    def _split(self, text: str) -> list[str]:
        if not text:
            return [""]
        chunks: list[str] = []
        remaining = text
        reopen_fence = False
        while len(remaining) + (4 if reopen_fence else 0) > self.max_length:
            prefix = "```\n" if reopen_fence else ""
            capacity = self.max_length - len(prefix) - 4
            cut = remaining.rfind("\n", 0, capacity + 1)
            if cut < capacity // 2:
                cut = remaining.rfind(" ", 0, capacity + 1)
            if cut < capacity // 2:
                cut = capacity
            raw = remaining[:cut].rstrip()
            remaining = remaining[cut:].lstrip()
            fence_open = (prefix + raw).count("```") % 2 == 1
            suffix = "\n```" if fence_open else ""
            chunks.append(prefix + raw + suffix)
            reopen_fence = fence_open
        if remaining:
            chunks.append(("```\n" if reopen_fence else "") + remaining)
        return chunks
