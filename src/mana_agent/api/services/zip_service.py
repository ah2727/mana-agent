from __future__ import annotations

import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable


class ZipValidationError(ValueError):
    """Raised when the upload does not satisfy the API ZIP contract."""


class InvalidZipError(ValueError):
    """Raised when uploaded bytes are not a valid ZIP archive."""


class UnsafeZipError(ValueError):
    """Raised when archive members could escape the extraction directory."""


def require_zip_filename(filename: str | None) -> str:
    clean = Path(str(filename or "")).name
    if not clean or clean == ".":
        raise ZipValidationError("Upload field 'file' is required.")
    if Path(clean).suffix.lower() != ".zip":
        raise ZipValidationError("Only .zip files are supported.")
    return clean


def _safe_member_path(name: str, extraction_dir: Path) -> Path:
    normalized = str(name or "").replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or any(part == ".." for part in pure.parts):
        raise UnsafeZipError("ZIP archive contains unsafe paths.")
    if not pure.parts or any(part in {"", "."} for part in pure.parts):
        raise UnsafeZipError("ZIP archive contains unsafe paths.")
    target = (extraction_dir / Path(*pure.parts)).resolve()
    root = extraction_dir.resolve()
    if target != root and root not in target.parents:
        raise UnsafeZipError("ZIP archive contains unsafe paths.")
    return target


def validate_zip_members(archive: zipfile.ZipFile, extraction_dir: Path) -> None:
    for info in archive.infolist():
        _safe_member_path(info.filename, extraction_dir)


def extract_zip_safely(zip_path: Path, extraction_dir: Path) -> None:
    extraction_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as archive:
            if archive.testzip() is not None:
                raise InvalidZipError("Uploaded file is not a valid ZIP archive.")
            validate_zip_members(archive, extraction_dir)
            for info in archive.infolist():
                target = _safe_member_path(info.filename, extraction_dir)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
    except zipfile.BadZipFile as exc:
        raise InvalidZipError("Uploaded file is not a valid ZIP archive.") from exc


def create_zip_from_directory(source_dir: Path, zip_path: Path, *, include: Iterable[str] | None = None) -> Path:
    names = set(include or [])
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(source_dir).as_posix()
            if names and rel not in names:
                continue
            archive.write(path, arcname=rel)
    return zip_path

