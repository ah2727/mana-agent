"""Document file detection, parsing, querying, and safe mutation helpers."""

from .detector import detect_document_type
from .service import DocumentService
from .types import DocumentChunk, DocumentFileType

__all__ = ["DocumentChunk", "DocumentFileType", "DocumentService", "detect_document_type"]
