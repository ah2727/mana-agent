from __future__ import annotations

from pathlib import Path
from typing import Any

from .cache import DocumentCache
from .detector import detect_document_type
from .query import query_chunks
from .readers import read_document
from .types import DocumentChunk, DocumentFileType
from .writers import create_document, delete_document, update_document


class DocumentService:
    def __init__(self, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.cache = DocumentCache(self.repo_root)

    def resolve_path(self, path: str | Path) -> Path:
        raw = Path(path)
        resolved = raw if raw.is_absolute() else self.repo_root / raw
        resolved = resolved.resolve()
        if self.repo_root not in resolved.parents and resolved != self.repo_root:
            raise ValueError("path escapes repository root")
        return resolved

    def detect(self, path: str, mime_type: str | None = None) -> dict[str, Any]:
        resolved = self.resolve_path(path)
        payload = detect_document_type(resolved, mime_type=mime_type).to_dict()
        payload["exists"] = resolved.exists()
        return {"ok": True, **payload}

    def discover(self, *, limit: int = 500) -> dict[str, Any]:
        files: list[dict[str, Any]] = []
        for path in sorted(self.repo_root.rglob("*")):
            if any(part in {".git", ".mana", ".venv", "venv", "node_modules", "__pycache__"} for part in path.relative_to(self.repo_root).parts):
                continue
            if not path.is_file():
                continue
            detected = detect_document_type(path)
            if detected.supported:
                row = detected.to_dict()
                row["path"] = path.relative_to(self.repo_root).as_posix()
                files.append(row)
            if len(files) >= limit:
                break
        return {"ok": True, "files": files, "count": len(files), "truncated": len(files) >= limit}

    def read(self, path: str, *, use_cache: bool = True, max_chunks: int = 400) -> dict[str, Any]:
        resolved = self.resolve_path(path)
        if not resolved.exists() or not resolved.is_file():
            return {"ok": False, "error": "file_not_found", "path": str(resolved)}
        if use_cache:
            cached, hit = self.cache.load(resolved)
            if hit and cached is not None:
                return {"ok": True, **cached["parsed"], "cache_hit": True}
        try:
            parsed = read_document(resolved, max_chunks=max_chunks).to_dict()
        except Exception as exc:
            return {"ok": False, "error": str(exc), "path": str(resolved), "detection": detect_document_type(resolved).to_dict()}
        self.cache.store(resolved, parsed)
        return {"ok": True, **parsed, "cache_hit": False}

    def analyze(self, path: str) -> dict[str, Any]:
        payload = self.read(path)
        if not payload.get("ok"):
            return payload
        chunks = payload.get("chunks") or []
        key_points = [str(item.get("content", "")).strip() for item in chunks[:5] if str(item.get("content", "")).strip()]
        tables = [item for item in chunks if item.get("kind") == "table"]
        return {
            "ok": True,
            "path": payload["path"],
            "file_type": payload["file_type"],
            "metadata": payload.get("metadata", {}),
            "analysis": payload.get("analysis", {}),
            "key_points": key_points,
            "tables": tables,
            "warnings": payload.get("warnings", []),
            "cache_hit": payload.get("cache_hit", False),
        }

    def query(self, query: str, *, paths: list[str] | None = None, file_types: list[str] | None = None, limit: int = 10, **filters: Any) -> dict[str, Any]:
        selected_paths = paths or [item["path"] for item in self.discover(limit=2000).get("files", [])]
        chunks: list[DocumentChunk] = []
        warnings: list[str] = []
        for raw in selected_paths:
            payload = self.read(str(raw))
            if not payload.get("ok"):
                warnings.append(f"{raw}: {payload.get('error')}")
                continue
            for row in payload.get("chunks", []):
                chunks.append(
                    DocumentChunk(
                        file_path=str(row.get("file_path", "")),
                        file_type=DocumentFileType(str(row.get("file_type"))),
                        content=str(row.get("content", "")),
                        chunk_id=str(row.get("chunk_id", "")),
                        section=str(row.get("section", "")),
                        page=row.get("page"),
                        sheet=str(row.get("sheet", "")),
                        row=row.get("row"),
                        column=row.get("column"),
                        kind=str(row.get("kind", "text")),
                        citation=dict(row.get("citation") or {}),
                    )
                )
        result = query_chunks(chunks, query, file_types=file_types, limit=limit, **filters)
        result["warnings"] = warnings
        return result

    def create(self, path: str, *, content: Any, file_type: str | None = None, overwrite: bool = False) -> dict[str, Any]:
        return create_document(self.resolve_path(path), content=content, file_type=file_type, overwrite=overwrite)

    def update(self, path: str, *, operation: str, payload: dict[str, Any], backup: bool = True) -> dict[str, Any]:
        return update_document(self.resolve_path(path), operation=operation, payload=payload, backup=backup)

    def delete(self, path: str, *, explicit: bool = False, backup: bool = True) -> dict[str, Any]:
        return delete_document(self.resolve_path(path), explicit=explicit, backup=backup)
