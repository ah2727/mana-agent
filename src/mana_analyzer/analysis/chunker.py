from __future__ import annotations

import logging

from mana_analyzer.analysis.models import CodeChunk, CodeSymbol

logger = logging.getLogger(__name__)


class CodeChunker:
    def __init__(self, max_chars: int = 1800, overlap: int = 200) -> None:
        self.max_chars = max_chars
        self.overlap = overlap

    def _chunk_text(self, text: str) -> list[str]:
        if len(text) <= self.max_chars:
            return [text]

        chunks: list[str] = []
        step = max(self.max_chars - self.overlap, 1)
        start = 0
        while start < len(text):
            end = start + self.max_chars
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start += step
        return chunks

    def build_chunks(self, symbols: list[CodeSymbol]) -> list[CodeChunk]:
        logger.debug("Building chunks for %d symbols", len(symbols))
        output: list[CodeChunk] = []
        for symbol in symbols:
            header = (
                f"kind: {symbol.kind}\n"
                f"name: {symbol.name}\n"
                f"signature: {symbol.signature}\n"
                f"file: {symbol.file_path}\n"
                f"line_range: {symbol.start_line}-{symbol.end_line}\n"
            )
            body = f"docstring:\n{symbol.docstring}\n\nsource:\n{symbol.source}"
            composed = f"{header}\n{body}"
            text_parts = self._chunk_text(composed)
            for idx, text in enumerate(text_parts):
                chunk_id = (
                    f"{symbol.file_path}:{symbol.start_line}:{symbol.end_line}:"
                    f"{symbol.kind}:{symbol.name}:{idx}"
                )
                output.append(
                    CodeChunk(
                        id=chunk_id,
                        text=text,
                        file_path=symbol.file_path,
                        start_line=symbol.start_line,
                        end_line=symbol.end_line,
                        symbol_name=symbol.name,
                        symbol_kind=symbol.kind,
                    )
                )
        logger.debug("Built %d chunks", len(output))
        return output
