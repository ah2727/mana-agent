"""
mana_analyzer.tools.write_file

A safe file-write tool for coding agents.

Key properties:
- Refuses path traversal / absolute paths.
- Refuses writing outside repo_root.
- Optionally restricts writes to allowed path prefixes (e.g. "src/", "tests/").
- Returns a JSON-serialisable result for tool-calling agents.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

# None => allow any path under repo_root
DEFAULT_ALLOWED_PREFIXES: Optional[tuple[str, ...]] = None


@dataclass(frozen=True)
class WriteFileResult:
    ok: bool
    path: str
    bytes_written: int = 0
    sha256: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalise_user_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _is_allowed_prefix(rel_posix: str, allowed_prefixes: Optional[Sequence[str]]) -> bool:
    if not allowed_prefixes:
        return True
    return any(rel_posix.startswith(prefix) for prefix in allowed_prefixes)


def safe_write_file(
    *,
    repo_root: Path,
    path: str,
    content: str,
    allowed_prefixes: Optional[Sequence[str]] = DEFAULT_ALLOWED_PREFIXES,
) -> dict[str, Any]:
    try:
        if "\x00" in path:
            return WriteFileResult(ok=False, path=path, error="Blocked: NUL byte in path").to_dict()

        user_path = Path(path)
        if user_path.is_absolute():
            return WriteFileResult(ok=False, path=path, error="Blocked: absolute paths are not allowed").to_dict()

        repo_root = repo_root.resolve()
        normalised = _normalise_user_path(path)
        target = (repo_root / normalised).resolve()

        try:
            rel = target.relative_to(repo_root)
        except ValueError:
            return WriteFileResult(ok=False, path=path, error="Blocked: path escapes repository root").to_dict()

        rel_posix = rel.as_posix()
        if not _is_allowed_prefix(rel_posix, allowed_prefixes):
            return WriteFileResult(
                ok=False,
                path=path,
                error=f"Blocked: writes restricted to prefixes {list(allowed_prefixes)}",
            ).to_dict()

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

        result = WriteFileResult(
            ok=True,
            path=rel_posix,
            bytes_written=len(content.encode("utf-8")),
            sha256=_sha256_text(content),
        )
        logger.info("Wrote file: %s (%d bytes)", rel_posix, result.bytes_written)
        return result.to_dict()

    except Exception as exc:  # noqa: BLE001
        logger.exception("write_file failed for %s", path)
        return WriteFileResult(ok=False, path=path, error=f"Error: {exc}").to_dict()


def build_write_file_tool(*, repo_root: Path, allowed_prefixes: Optional[Sequence[str]] = DEFAULT_ALLOWED_PREFIXES):
    try:
        from langchain_core.tools import StructuredTool  # type: ignore
    except Exception:  # pragma: no cover
        from langchain.tools import StructuredTool  # type: ignore

    def _tool(path: str, content: str) -> dict[str, Any]:
        return safe_write_file(repo_root=repo_root, path=path, content=content, allowed_prefixes=allowed_prefixes)

    return StructuredTool.from_function(
        func=_tool,
        name="write_file",
        description=(
            "Safely write text content to a file under the repository root. "
            "Refuses absolute paths, path traversal, and paths escaping the repository root."
        ),
    )