from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class DocumentFileType(str, Enum):
    DOCX = "docx"
    PDF = "pdf"
    XLSX = "xlsx"
    XLSM = "xlsm"
    CSV = "csv"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class DocumentDetection:
    path: str
    file_type: DocumentFileType
    extension: str
    mime_type: str
    supported: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["file_type"] = self.file_type.value
        return data


@dataclass(frozen=True)
class DocumentChunk:
    file_path: str
    file_type: DocumentFileType
    content: str
    chunk_id: str
    section: str = ""
    page: int | None = None
    sheet: str = ""
    row: int | None = None
    column: int | None = None
    kind: str = "text"
    citation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["file_type"] = self.file_type.value
        return data


@dataclass(frozen=True)
class ParsedDocument:
    path: str
    file_type: DocumentFileType
    metadata: dict[str, Any]
    chunks: list[DocumentChunk]
    analysis: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "file_type": self.file_type.value,
            "metadata": self.metadata,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "analysis": self.analysis,
            "warnings": self.warnings,
        }


def chunk_id_for(path: Path, location: str) -> str:
    safe = str(path).replace("\\", "/")
    return f"{safe}#{location}"
