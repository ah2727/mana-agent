from __future__ import annotations

from copy import deepcopy
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
        ignored_dirs = {".git", ".mana", ".venv", "venv", "node_modules", "__pycache__"}

        for path in sorted(self.repo_root.rglob("*")):
            relative_parts = path.relative_to(self.repo_root).parts
            if any(part in ignored_dirs for part in relative_parts):
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

        return {
            "ok": True,
            "files": files,
            "count": len(files),
            "truncated": len(files) >= limit,
        }

    def read(
        self,
        path: str,
        *,
        use_cache: bool = True,
        max_chunks: int = 400,
    ) -> dict[str, Any]:
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
            return {
                "ok": False,
                "error": str(exc),
                "path": str(resolved),
                "detection": detect_document_type(resolved).to_dict(),
            }

        self.cache.store(resolved, parsed)
        return {"ok": True, **parsed, "cache_hit": False}

    def analyze(self, path: str) -> dict[str, Any]:
        payload = self.read(path)
        if not payload.get("ok"):
            return payload

        chunks = payload.get("chunks") or []
        key_points = [
            str(item.get("content", "")).strip()
            for item in chunks[:5]
            if str(item.get("content", "")).strip()
        ]
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

    def query(
        self,
        query: str,
        *,
        paths: list[str] | None = None,
        file_types: list[str] | None = None,
        limit: int = 10,
        **filters: Any,
    ) -> dict[str, Any]:
        selected_paths = paths or [
            item["path"] for item in self.discover(limit=2000).get("files", [])
        ]

        chunks: list[DocumentChunk] = []
        warnings: list[str] = []

        for raw in selected_paths:
            payload = self.read(str(raw))
            if not payload.get("ok"):
                warnings.append(f"{raw}: {payload.get('error')}")
                continue

            for row in payload.get("chunks", []):
                try:
                    file_type = DocumentFileType(str(row.get("file_type")))
                except ValueError:
                    warnings.append(
                        f"{raw}: unsupported chunk file_type={row.get('file_type')!r}"
                    )
                    continue

                chunks.append(
                    DocumentChunk(
                        file_path=str(row.get("file_path", "")),
                        file_type=file_type,
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

        result = query_chunks(
            chunks,
            query,
            file_types=file_types,
            limit=limit,
            **filters,
        )
        result["warnings"] = warnings
        return result

    def create(
        self,
        path: str,
        *,
        content: Any,
        file_type: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        resolved = self.resolve_path(path)

        normalized_content = self._normalize_create_content(
            resolved,
            content=content,
            file_type=file_type,
        )

        result = create_document(
            resolved,
            content=normalized_content,
            file_type=file_type,
            overwrite=overwrite,
        )

        self._invalidate_cache_safely(resolved)
        return result

    def update(
        self,
        path: str,
        *,
        operation: str,
        payload: dict[str, Any],
        backup: bool = True,
    ) -> dict[str, Any]:
        resolved = self.resolve_path(path)

        result = update_document(
            resolved,
            operation=operation,
            payload=payload,
            backup=backup,
        )

        self._invalidate_cache_safely(resolved)
        return result

    def delete(
        self,
        path: str,
        *,
        explicit: bool = False,
        backup: bool = True,
    ) -> dict[str, Any]:
        resolved = self.resolve_path(path)

        result = delete_document(
            resolved,
            explicit=explicit,
            backup=backup,
        )

        self._invalidate_cache_safely(resolved)
        return result

    def _normalize_create_content(
        self,
        path: Path,
        *,
        content: Any,
        file_type: str | None,
    ) -> Any:
        """
        Normalize document_create payloads before passing them to writers.

        Main Excel fix:
        Old/bad payloads may look like this:

            {
                "sheets": {
                    "Sheet1": [
                        {"type": "table", ...},
                        {"type": "formula", ...}
                    ]
                }
            }

        Many writers expect:

            {
                "sheets": {
                    "Sheet1": {
                        "tables": [...],
                        "cells": [...]
                    }
                }
            }

        Without this normalization, the writer can create only the sheet name
        and ignore the actual table/formula blocks.
        """
        document_type = self._resolve_document_file_type(path, file_type)

        if document_type not in {
            DocumentFileType.XLSX,
            DocumentFileType.XLSM,
        }:
            return content

        if not isinstance(content, dict):
            return content

        normalized = deepcopy(content)

        sheets = normalized.get("sheets")
        if not isinstance(sheets, dict):
            return normalized

        normalized_sheets: dict[str, Any] = {}

        for sheet_name, sheet_spec in sheets.items():
            normalized_sheets[str(sheet_name)] = self._normalize_excel_sheet_spec(
                sheet_spec
            )

        normalized["sheets"] = normalized_sheets
        return normalized

    def _resolve_document_file_type(
        self,
        path: Path,
        file_type: str | None,
    ) -> DocumentFileType | None:
        if file_type:
            try:
                return DocumentFileType(str(file_type).lower())
            except ValueError:
                return None

        detected = detect_document_type(path)

        raw_file_type = getattr(detected, "file_type", None)
        if isinstance(raw_file_type, DocumentFileType):
            return raw_file_type

        if raw_file_type:
            try:
                return DocumentFileType(str(raw_file_type).lower())
            except ValueError:
                return None

        suffix = path.suffix.lower()
        if suffix == ".xlsx":
            return DocumentFileType.XLSX
        if suffix == ".xlsm":
            return DocumentFileType.XLSM

        return None

    def _normalize_excel_sheet_spec(self, sheet_spec: Any) -> Any:
        """
        Supports both accepted schemas:

        1. New/dict schema:
            {
                "cells": [{"cell": "A1", "value": 123}],
                "tables": [{"start_cell": "A1", "columns": [...], "rows": [...]}]
            }

        2. Legacy/list schema:
            [
                {"type": "table", "start_cell": "A1", "columns": [...], "rows": [...]},
                {"type": "formula", "cell": "A5", "formula": "=SUM(A1:A4)"}
            ]
        """
        if isinstance(sheet_spec, list):
            return self._normalize_excel_sheet_blocks(sheet_spec)

        if isinstance(sheet_spec, dict):
            normalized = deepcopy(sheet_spec)

            cells = list(normalized.get("cells") or [])
            tables = list(normalized.get("tables") or [])

            # Support a single table directly on the sheet spec.
            if self._looks_like_table_block(normalized):
                table = self._normalize_table_block(normalized)
                if table:
                    tables.append(table)

            # Support a single cell/formula directly on the sheet spec.
            if self._looks_like_cell_block(normalized):
                cell = self._normalize_cell_block(normalized)
                if cell:
                    cells.append(cell)

            # Support formulas as a separate collection, but also convert them
            # into cells because most openpyxl writers just assign formulas
            # through normal cell writes.
            formulas = normalized.get("formulas") or []
            if isinstance(formulas, list):
                for formula_block in formulas:
                    if isinstance(formula_block, dict):
                        cell = self._normalize_formula_block(formula_block)
                        if cell:
                            cells.append(cell)

            if cells:
                normalized["cells"] = cells
            if tables:
                normalized["tables"] = tables

            return normalized

        return sheet_spec

    def _normalize_excel_sheet_blocks(
        self,
        blocks: list[Any],
    ) -> dict[str, Any]:
        cells: list[dict[str, Any]] = []
        tables: list[dict[str, Any]] = []
        unknown_blocks: list[Any] = []

        for block in blocks:
            if not isinstance(block, dict):
                unknown_blocks.append(block)
                continue

            block_type = str(block.get("type", "")).lower().strip()

            if block_type == "table" or self._looks_like_table_block(block):
                table = self._normalize_table_block(block)
                if table:
                    tables.append(table)
                continue

            if block_type == "formula":
                cell = self._normalize_formula_block(block)
                if cell:
                    cells.append(cell)
                continue

            if block_type == "cell" or self._looks_like_cell_block(block):
                cell = self._normalize_cell_block(block)
                if cell:
                    cells.append(cell)
                continue

            if block_type == "cells" and isinstance(block.get("cells"), list):
                for item in block["cells"]:
                    if isinstance(item, dict):
                        cell = self._normalize_cell_block(item)
                        if cell:
                            cells.append(cell)
                continue

            if block_type == "tables" and isinstance(block.get("tables"), list):
                for item in block["tables"]:
                    if isinstance(item, dict):
                        table = self._normalize_table_block(item)
                        if table:
                            tables.append(table)
                continue

            unknown_blocks.append(block)

        normalized: dict[str, Any] = {}
        if cells:
            normalized["cells"] = cells
        if tables:
            normalized["tables"] = tables
        if unknown_blocks:
            normalized["unknown_blocks"] = unknown_blocks

        return normalized

    def _looks_like_table_block(self, block: dict[str, Any]) -> bool:
        return (
            "rows" in block
            and isinstance(block.get("rows"), list)
            and (
                "columns" in block
                or "headers" in block
                or "start_cell" in block
            )
        )

    def _looks_like_cell_block(self, block: dict[str, Any]) -> bool:
        return "cell" in block and (
            "value" in block
            or "formula" in block
        )

    def _normalize_table_block(
        self,
        block: dict[str, Any],
    ) -> dict[str, Any] | None:
        rows = block.get("rows")
        if not isinstance(rows, list):
            return None

        table: dict[str, Any] = {
            "start_cell": str(block.get("start_cell") or "A1"),
            "rows": rows,
        }

        columns = block.get("columns", block.get("headers"))
        if isinstance(columns, list):
            table["columns"] = columns

        if "name" in block:
            table["name"] = block["name"]

        if "style" in block:
            table["style"] = block["style"]

        return table

    def _normalize_formula_block(
        self,
        block: dict[str, Any],
    ) -> dict[str, Any] | None:
        cell = block.get("cell")
        formula = block.get("formula")

        if not cell or formula is None:
            return None

        return {
            "cell": str(cell),
            "formula": str(formula),
        }

    def _normalize_cell_block(
        self,
        block: dict[str, Any],
    ) -> dict[str, Any] | None:
        cell = block.get("cell")
        if not cell:
            return None

        normalized: dict[str, Any] = {"cell": str(cell)}

        if "formula" in block:
            normalized["formula"] = str(block["formula"])
        elif "value" in block:
            normalized["value"] = block["value"]
        else:
            normalized["value"] = None

        return normalized

    def _invalidate_cache_safely(self, path: Path) -> None:
        """
        Keep this defensive because DocumentCache implementation may expose
        different invalidation method names.
        """
        for method_name in (
            "invalidate",
            "delete",
            "remove",
            "clear_path",
            "drop",
        ):
            method = getattr(self.cache, method_name, None)
            if callable(method):
                try:
                    method(path)
                except TypeError:
                    method(str(path))
                return