from __future__ import annotations

import mimetypes
from pathlib import Path

from .types import DocumentDetection, DocumentFileType

_EXTENSIONS = {
    ".docx": DocumentFileType.DOCX,
    ".pdf": DocumentFileType.PDF,
    ".xlsx": DocumentFileType.XLSX,
    ".xlsm": DocumentFileType.XLSM,
    ".csv": DocumentFileType.CSV,
}

_MIME_TYPES = {
    "application/pdf": DocumentFileType.PDF,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocumentFileType.DOCX,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": DocumentFileType.XLSX,
    "application/vnd.ms-excel.sheet.macroenabled.12": DocumentFileType.XLSM,
    "text/csv": DocumentFileType.CSV,
    "application/csv": DocumentFileType.CSV,
}


def detect_document_type(path: str | Path, mime_type: str | None = None) -> DocumentDetection:
    resolved = Path(path)
    extension = resolved.suffix.lower()
    guessed_mime = mime_type or mimetypes.guess_type(str(resolved))[0] or ""
    file_type = _MIME_TYPES.get(guessed_mime) or _EXTENSIONS.get(extension, DocumentFileType.UNSUPPORTED)
    supported = file_type != DocumentFileType.UNSUPPORTED
    reason = "" if supported else f"unsupported document extension or MIME type: {extension or guessed_mime or 'unknown'}"
    return DocumentDetection(
        path=str(resolved),
        file_type=file_type,
        extension=extension,
        mime_type=guessed_mime,
        supported=supported,
        reason=reason,
    )
