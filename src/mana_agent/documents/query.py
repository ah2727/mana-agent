from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .types import DocumentChunk


def query_chunks(
    chunks: list[DocumentChunk],
    query: str,
    *,
    file_types: list[str] | None = None,
    path_filter: str = "",
    sheet: str = "",
    page: int | None = None,
    section: str = "",
    limit: int = 10,
) -> dict[str, Any]:
    terms = set(re.findall(r"[a-zA-Z0-9_]+", str(query).lower()))
    max_items = max(1, min(int(limit or 10), 100))
    allowed_types = {item.lower().lstrip(".") for item in file_types or [] if item}
    scored: list[tuple[float, DocumentChunk]] = []
    for chunk in chunks:
        if allowed_types and chunk.file_type.value not in allowed_types:
            continue
        if path_filter and path_filter not in Path(chunk.file_path).as_posix():
            continue
        if sheet and chunk.sheet != sheet:
            continue
        if page is not None and chunk.page != page:
            continue
        if section and section.lower() not in chunk.section.lower():
            continue
        content_terms = set(re.findall(r"[a-zA-Z0-9_]+", chunk.content.lower()))
        overlap = terms & content_terms if terms else set()
        if terms and not overlap and str(query).lower() not in chunk.content.lower():
            continue
        score = len(overlap) / max(len(terms), 1) if terms else 1.0
        scored.append((float(score), chunk))
    results = [
        {
            **chunk.to_dict(),
            "score": score,
            "snippet": chunk.content[:500],
        }
        for score, chunk in sorted(scored, key=lambda item: (-item[0], item[1].chunk_id))[:max_items]
    ]
    return {"ok": True, "query": query, "results": results, "count": len(results)}
