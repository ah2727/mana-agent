from __future__ import annotations

import csv
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .detector import detect_document_type
from .types import DocumentFileType


def _require(module_name: str, package: str) -> Any:
    try:
        return __import__(module_name, fromlist=["*"])
    except ImportError as exc:
        raise RuntimeError(f"{package} is required for this document operation") from exc


def _backup(path: Path) -> str:
    backup_path = path.with_suffix(path.suffix + ".bak")
    counter = 1
    while backup_path.exists():
        backup_path = path.with_suffix(path.suffix + f".bak{counter}")
        counter += 1
    shutil.copy2(path, backup_path)
    return str(backup_path)


def _atomic_save(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent), suffix=path.suffix) as tmp:
        temp_path = Path(tmp.name)
    try:
        writer(temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def create_document(path: Path, *, content: Any, file_type: str | None = None, overwrite: bool = False) -> dict[str, Any]:
    if path.exists() and not overwrite:
        return {"ok": False, "error": "target_exists", "path": str(path)}
    detected = detect_document_type(path)
    kind = DocumentFileType(file_type) if file_type else detected.file_type
    if kind == DocumentFileType.DOCX:
        docx = _require("docx", "python-docx")
        doc = docx.Document()
        payload = content if isinstance(content, dict) else {"paragraphs": str(content).splitlines()}
        title = str(payload.get("title") or "").strip()
        if title:
            doc.add_heading(title, level=1)
        for paragraph in payload.get("paragraphs") or []:
            text = str(paragraph).strip()
            if text:
                doc.add_paragraph(text)
        for table_payload in payload.get("tables") or []:
            rows = list(table_payload or [])
            if not rows:
                continue
            table = doc.add_table(rows=len(rows), cols=max(len(row) for row in rows))
            for row_index, row in enumerate(rows):
                for col_index, value in enumerate(row):
                    table.cell(row_index, col_index).text = str(value)
        _atomic_save(path, lambda target: doc.save(str(target)))
        return {"ok": True, "path": str(path), "file_type": kind.value, "created": True}
    if kind in {DocumentFileType.XLSX, DocumentFileType.XLSM}:
        openpyxl = _require("openpyxl", "openpyxl")
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Sheet1"
        rows = content.get("rows", content) if isinstance(content, dict) else content
        if rows and isinstance(rows, list) and isinstance(rows[0], dict):
            headers = list(rows[0].keys())
            sheet.append(headers)
            for row in rows:
                sheet.append([row.get(header) for header in headers])
        else:
            for row in rows or []:
                sheet.append(list(row) if isinstance(row, (list, tuple)) else [row])
        _atomic_save(path, lambda target: workbook.save(str(target)))
        return {"ok": True, "path": str(path), "file_type": kind.value, "created": True}
    if kind == DocumentFileType.CSV:
        rows = content.get("rows", content) if isinstance(content, dict) else content
        def write_csv(target: Path) -> None:
            with target.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                if rows and isinstance(rows, list) and isinstance(rows[0], dict):
                    headers = list(rows[0].keys())
                    writer.writerow(headers)
                    for row in rows:
                        writer.writerow([row.get(header) for header in headers])
                else:
                    writer.writerows(rows or [])
        _atomic_save(path, write_csv)
        return {"ok": True, "path": str(path), "file_type": kind.value, "created": True}
    if kind == DocumentFileType.PDF:
        text = content.get("text", "") if isinstance(content, dict) else str(content)
        _write_simple_text_pdf(path, text)
        return {"ok": True, "path": str(path), "file_type": kind.value, "created": True}
    return {"ok": False, "error": "unsupported_file_type", "path": str(path), "file_type": kind.value}


def update_document(path: Path, *, operation: str, payload: dict[str, Any], backup: bool = True) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "error": "file_not_found", "path": str(path)}
    kind = detect_document_type(path).file_type
    backup_path = _backup(path) if backup else ""
    if kind == DocumentFileType.DOCX:
        return _update_docx(path, operation=operation, payload=payload, backup_path=backup_path)
    if kind in {DocumentFileType.XLSX, DocumentFileType.XLSM}:
        return _update_workbook(path, operation=operation, payload=payload, backup_path=backup_path, keep_vba=kind == DocumentFileType.XLSM)
    if kind == DocumentFileType.PDF and operation == "metadata":
        pypdf = _require("pypdf", "pypdf")
        reader = pypdf.PdfReader(str(path))
        writer = pypdf.PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        writer.add_metadata({str(key): str(value) for key, value in payload.items()})
        def write_pdf(target: Path) -> None:
            with target.open("wb") as handle:
                writer.write(handle)

        _atomic_save(path, write_pdf)
        return {"ok": True, "path": str(path), "operation": operation, "backup_path": backup_path}
    return {"ok": False, "error": "unsupported_update_operation", "path": str(path), "operation": operation, "backup_path": backup_path}


def delete_document(path: Path, *, explicit: bool = False, backup: bool = True) -> dict[str, Any]:
    if not explicit:
        return {"ok": False, "error": "explicit_delete_required", "path": str(path)}
    if not path.exists() or not path.is_file():
        return {"ok": False, "error": "file_not_found", "path": str(path)}
    backup_path = _backup(path) if backup else ""
    path.unlink()
    return {"ok": True, "path": str(path), "deleted": True, "backup_path": backup_path}


def _update_docx(path: Path, *, operation: str, payload: dict[str, Any], backup_path: str) -> dict[str, Any]:
    docx = _require("docx", "python-docx")
    doc = docx.Document(str(path))
    if operation == "append_section":
        title = str(payload.get("title") or "").strip()
        if title:
            doc.add_heading(title, level=int(payload.get("level") or 1))
        for paragraph in payload.get("paragraphs") or [payload.get("text", "")]:
            text = str(paragraph).strip()
            if text:
                doc.add_paragraph(text)
    elif operation == "replace_text":
        old = str(payload.get("old_text") or "")
        new = str(payload.get("new_text") or "")
        if not old:
            return {"ok": False, "error": "old_text_required", "path": str(path), "backup_path": backup_path}
        replaced = 0
        for paragraph in doc.paragraphs:
            if old in paragraph.text:
                paragraph.text = paragraph.text.replace(old, new)
                replaced += 1
        if replaced == 0:
            return {"ok": False, "error": "text_not_found", "path": str(path), "backup_path": backup_path}
    elif operation == "add_table":
        rows = list(payload.get("rows") or [])
        if not rows:
            return {"ok": False, "error": "rows_required", "path": str(path), "backup_path": backup_path}
        table = doc.add_table(rows=len(rows), cols=max(len(row) for row in rows))
        for row_index, row in enumerate(rows):
            for col_index, value in enumerate(row):
                table.cell(row_index, col_index).text = str(value)
    elif operation == "metadata":
        props = doc.core_properties
        for key, value in payload.items():
            if hasattr(props, key):
                setattr(props, key, str(value))
    else:
        return {"ok": False, "error": "unsupported_docx_operation", "path": str(path), "backup_path": backup_path}
    _atomic_save(path, lambda target: doc.save(str(target)))
    return {"ok": True, "path": str(path), "operation": operation, "backup_path": backup_path}


def _update_workbook(path: Path, *, operation: str, payload: dict[str, Any], backup_path: str, keep_vba: bool) -> dict[str, Any]:
    openpyxl = _require("openpyxl", "openpyxl")
    workbook = openpyxl.load_workbook(str(path), data_only=False, keep_vba=keep_vba)
    if operation == "update_cell":
        sheet = workbook[str(payload.get("sheet") or workbook.sheetnames[0])]
        coordinate = str(payload.get("cell") or "")
        if not coordinate:
            return {"ok": False, "error": "cell_required", "path": str(path), "backup_path": backup_path}
        current = sheet[coordinate].value
        if isinstance(current, str) and current.startswith("=") and not bool(payload.get("replace_formula", False)):
            return {"ok": False, "error": "formula_replacement_requires_explicit_flag", "path": str(path), "backup_path": backup_path}
        sheet[coordinate] = payload.get("value")
    elif operation == "append_rows":
        sheet = workbook[str(payload.get("sheet") or workbook.sheetnames[0])]
        for row in payload.get("rows") or []:
            sheet.append(row)
    elif operation == "create_sheet":
        name = str(payload.get("sheet") or "").strip()
        if not name:
            return {"ok": False, "error": "sheet_required", "path": str(path), "backup_path": backup_path}
        workbook.create_sheet(title=name)
    elif operation == "rename_sheet":
        old = str(payload.get("sheet") or "")
        new = str(payload.get("new_name") or "")
        workbook[old].title = new
    elif operation == "delete_rows":
        sheet = workbook[str(payload.get("sheet") or workbook.sheetnames[0])]
        sheet.delete_rows(int(payload.get("idx") or 1), int(payload.get("amount") or 1))
    else:
        return {"ok": False, "error": "unsupported_workbook_operation", "path": str(path), "backup_path": backup_path}
    _atomic_save(path, lambda target: workbook.save(str(target)))
    return {
        "ok": True,
        "path": str(path),
        "operation": operation,
        "backup_path": backup_path,
        "warning": "Macros preserved with keep_vba=True; verify workbook macros after editing." if keep_vba else "",
    }


def _write_simple_text_pdf(path: Path, text: str) -> None:
    safe = str(text or "").replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content = f"BT /F1 12 Tf 72 720 Td ({safe[:3000]}) Tj ET".encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(b"%PDF-1.4\n")
        offsets = [0]
        for index, obj in enumerate(objects, start=1):
            offsets.append(handle.tell())
            handle.write(f"{index} 0 obj\n".encode("ascii") + obj + b"\nendobj\n")
        xref = handle.tell()
        handle.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
        for offset in offsets[1:]:
            handle.write(f"{offset:010d} 00000 n \n".encode("ascii"))
        handle.write(
            f"trailer << /Root 1 0 R /Size {len(objects) + 1} >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
        )
