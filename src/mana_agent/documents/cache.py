from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mana_agent.utils.io import ensure_dir


class DocumentCache:
    def __init__(self, repo_root: Path, cache_dir: Path | None = None) -> None:
        self.repo_root = repo_root.resolve()
        self.cache_dir = ensure_dir(cache_dir or (self.repo_root / ".mana" / "document_cache"))

    def fingerprint(self, path: Path) -> dict[str, Any]:
        stat = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "path": str(path.resolve()),
            "mtime_ns": int(stat.st_mtime_ns),
            "size": int(stat.st_size),
            "sha256": digest,
        }

    def _cache_path(self, fingerprint: dict[str, Any]) -> Path:
        key = hashlib.sha256(
            json.dumps(fingerprint, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{key}.json"

    def load(self, path: Path) -> tuple[dict[str, Any] | None, bool]:
        fp = self.fingerprint(path)
        cache_path = self._cache_path(fp)
        if not cache_path.exists():
            return None, False
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return None, False
        if payload.get("fingerprint") != fp:
            return None, False
        payload["cache_hit"] = True
        return payload, True

    def store(self, path: Path, parsed: dict[str, Any]) -> dict[str, Any]:
        fp = self.fingerprint(path)
        payload = {"fingerprint": fp, "parsed": parsed, "cache_hit": False}
        self._cache_path(fp).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return payload
