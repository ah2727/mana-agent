from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .detector import detect_document_type
from .types import DocumentChunk, DocumentFileType, ParsedDocument, chunk_id_for


def _require(module_name: str, package: str) -> Any:
    try:
        return __import__(module_name, fromlist=["*"])
    except ImportError as exc:
        raise RuntimeError(f"{package} is required for this document operation") from exc


def read_document(path: str | Path, *, max_chunks: int = 400, max_chars_per_chunk: int = 4000) -> ParsedDocument:
    resolved = Path(path).resolve()
    detection = detect_document_type(resolved)
    if detection.file_type == DocumentFileType.DOCX:
        return read_docx(resolved, max_chunks=max_chunks, max_chars_per_chunk=max_chars_per_chunk)
    if detection.file_type == DocumentFileType.PDF:
        return read_pdf(resolved, max_chunks=max_chunks, max_chars_per_chunk=max_chars_per_chunk)
    if detection.file_type in {DocumentFileType.XLSX, DocumentFileType.XLSM}:
        return read_workbook(resolved, file_type=detection.file_type, max_chunks=max_chunks)
    if detection.file_type == DocumentFileType.CSV:
        return read_csv(resolved, max_chunks=max_chunks)
    raise ValueError(detection.reason)


def read_docx(path: Path, *, max_chunks: int, max_chars_per_chunk: int) -> ParsedDocument:
    docx = _require("docx", "python-docx")
    doc = docx.Document(str(path))
    chunks: list[DocumentChunk] = []
    headings = 0
    paragraph_count = 0
    for index, paragraph in enumerate(doc.paragraphs, start=1):
        text = paragraph.text.strip()
        if not text:
            continue
        style = getattr(getattr(paragraph, "style", None), "name", "") or ""
        kind = "heading" if style.lower().startswith("heading") else "paragraph"
        headings += int(kind == "heading")
        paragraph_count += 1
        chunks.append(
            DocumentChunk(
                file_path=str(path),
                file_type=DocumentFileType.DOCX,
                section=style or f"paragraph {index}",
                kind=kind,
                content=text[:max_chars_per_chunk],
                chunk_id=chunk_id_for(path, f"paragraph-{index}"),
                citation={"paragraph": index, "style": style},
            )
        )
        if len(chunks) >= max_chunks:
            break
    for table_index, table in enumerate(doc.tables, start=1):
        rows = []
        for row in table.rows:
            rows.append([cell.text.strip() for cell in row.cells])
        text = "\n".join(" | ".join(cell for cell in row) for row in rows if any(row)).strip()
        if text:
            chunks.append(
                DocumentChunk(
                    file_path=str(path),
                    file_type=DocumentFileType.DOCX,
                    section=f"table {table_index}",
                    kind="table",
                    content=text[:max_chars_per_chunk],
                    chunk_id=chunk_id_for(path, f"table-{table_index}"),
                    citation={"table": table_index},
                )
            )
        if len(chunks) >= max_chunks:
            break
    props = doc.core_properties
    metadata = {
        "title": props.title or "",
        "author": props.author or "",
        "subject": props.subject or "",
        "created": props.created.isoformat() if props.created else "",
        "modified": props.modified.isoformat() if props.modified else "",
    }
    return ParsedDocument(
        path=str(path),
        file_type=DocumentFileType.DOCX,
        metadata=metadata,
        chunks=chunks,
        analysis={
            "paragraph_count": paragraph_count,
            "heading_count": headings,
            "table_count": len(doc.tables),
            "chunk_count": len(chunks),
            "empty": not chunks,
        },
        warnings=[] if chunks else ["No readable text found in DOCX."],
    )


def read_pdf(path: Path, *, max_chunks: int, max_chars_per_chunk: int) -> ParsedDocument:
    pypdf = _require("pypdf", "pypdf")
    reader = pypdf.PdfReader(str(path))
    chunks: list[DocumentChunk] = []
    empty_pages = 0
    for page_index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            empty_pages += 1
            continue
        chunks.append(
            DocumentChunk(
                file_path=str(path),
                file_type=DocumentFileType.PDF,
                page=page_index,
                section=f"page {page_index}",
                kind="page",
                content=text[:max_chars_per_chunk],
                chunk_id=chunk_id_for(path, f"page-{page_index}"),
                citation={"page": page_index},
            )
        )
        if len(chunks) >= max_chunks:
            break
    metadata = {str(key).lstrip("/"): str(value) for key, value in (reader.metadata or {}).items()}
    scanned = len(chunks) == 0 and len(reader.pages) > 0
    warnings = ["PDF appears scanned or image-only; OCR is required for text extraction."] if scanned else []
    return ParsedDocument(
        path=str(path),
        file_type=DocumentFileType.PDF,
        metadata=metadata,
        chunks=chunks,
        analysis={
            "page_count": len(reader.pages),
            "readable_page_count": len(chunks),
            "empty_page_count": empty_pages,
            "chunk_count": len(chunks),
            "needs_ocr": scanned,
            "empty": not chunks,
            "table_extraction": "not_available",
        },
        warnings=warnings,
    )


def read_workbook(path: Path, *, file_type: DocumentFileType, max_chunks: int) -> ParsedDocument:
    openpyxl = _require("openpyxl", "openpyxl")
    workbook = openpyxl.load_workbook(str(path), data_only=False, read_only=False, keep_vba=file_type == DocumentFileType.XLSM)
    chunks: list[DocumentChunk] = []
    sheet_summaries: list[dict[str, Any]] = []
    formula_count = 0
    for sheet in workbook.worksheets:
        headers: list[str] = []
        rows_with_values = 0
        empty_rows = 0
        empty_columns = 0
        for row_index, row in enumerate(sheet.iter_rows(), start=1):
            values = [cell.value for cell in row]
            if not any(value not in (None, "") for value in values):
                empty_rows += 1
                continue
            rows_with_values += 1
            if row_index == 1:
                headers = [str(value) for value in values if value not in (None, "")]
            text_parts = []
            for cell in row:
                value = cell.value
                if value in (None, ""):
                    continue
                if isinstance(value, str) and value.startswith("="):
                    formula_count += 1
                text_parts.append(f"{cell.coordinate}={value}")
            if text_parts and len(chunks) < max_chunks:
                chunks.append(
                    DocumentChunk(
                        file_path=str(path),
                        file_type=file_type,
                        sheet=sheet.title,
                        row=row_index,
                        section=sheet.title,
                        kind="row",
                        content="; ".join(text_parts),
                        chunk_id=chunk_id_for(path, f"{sheet.title}-row-{row_index}"),
                        citation={"sheet": sheet.title, "row": row_index},
                    )
                )
        for column in sheet.iter_cols():
            if not any(cell.value not in (None, "") for cell in column):
                empty_columns += 1
        sheet_summaries.append(
            {
                "name": sheet.title,
                "dimensions": sheet.calculate_dimension(),
                "max_row": sheet.max_row,
                "max_column": sheet.max_column,
                "headers": headers,
                "rows_with_values": rows_with_values,
                "empty_rows": empty_rows,
                "empty_columns": empty_columns,
            }
        )
    metadata = {
        "sheet_names": workbook.sheetnames,
        "properties": {
            key: str(value)
            for key, value in workbook.properties.__dict__.items()
            if not key.startswith("_") and value not in (None, "")
        },
    }
    return ParsedDocument(
        path=str(path),
        file_type=file_type,
        metadata=metadata,
        chunks=chunks,
        analysis={
            "sheet_count": len(workbook.worksheets),
            "sheets": sheet_summaries,
            "formula_count": formula_count,
            "chunk_count": len(chunks),
            "empty": not chunks,
            "macros_preserved": file_type == DocumentFileType.XLSM,
        },
        warnings=["Macro-enabled workbook loaded with keep_vba=True. Avoid operations that rewrite macros."] if file_type == DocumentFileType.XLSM else [],
    )


def read_csv(path: Path, *, max_chunks: int) -> ParsedDocument:
    chunks: list[DocumentChunk] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        headers: list[str] = []
        for row_index, row in enumerate(reader, start=1):
            if row_index == 1:
                headers = [str(item) for item in row]
            if any(str(item).strip() for item in row):
                chunks.append(
                    DocumentChunk(
                        file_path=str(path),
                        file_type=DocumentFileType.CSV,
                        section="csv",
                        row=row_index,
                        kind="row",
                        content=" | ".join(row),
                        chunk_id=chunk_id_for(path, f"row-{row_index}"),
                        citation={"row": row_index},
                    )
                )
            if len(chunks) >= max_chunks:
                break
    return ParsedDocument(
        path=str(path),
        file_type=DocumentFileType.CSV,
        metadata={"headers": headers},
        chunks=chunks,
        analysis={"row_count": len(chunks), "headers": headers, "chunk_count": len(chunks), "empty": not chunks},
        warnings=[],
    )
