from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
import shutil

from mana_analyzer.analysis.chunker import CodeChunker
from mana_analyzer.analysis.models import CodeChunk
from mana_analyzer.parsers.python_parser import PythonParser
from mana_analyzer.utils.io import (
    ensure_dir,
    iter_source_files,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from mana_analyzer.vector_store.faiss_store import FaissStore

logger = logging.getLogger(__name__)


class IndexService:
    def __init__(self, parser: PythonParser, chunker: CodeChunker, store: FaissStore) -> None:
        self.parser = parser
        self.chunker = chunker
        self.store = store

    @staticmethod
    def _manifest_path(index_dir: Path) -> Path:
        return index_dir / "manifest.json"

    @staticmethod
    def _chunks_path(index_dir: Path) -> Path:
        return index_dir / "chunks.jsonl"

    def _load_manifest(self, index_dir: Path) -> dict:
        logger.debug("Loading manifest from %s", self._manifest_path(index_dir))
        payload = read_json(self._manifest_path(index_dir))
        if not payload:
            payload = {"files": {}}
        payload.setdefault("files", {})
        return payload

    def _load_chunks(self, index_dir: Path) -> dict[str, CodeChunk]:
        logger.debug("Loading chunks from %s", self._chunks_path(index_dir))
        rows = read_jsonl(self._chunks_path(index_dir))
        chunks: dict[str, CodeChunk] = {}
        for row in rows:
            chunk = CodeChunk(**row)
            chunks[chunk.id] = chunk
        return chunks

    def index(self, target_path: str | Path, index_dir: str | Path, rebuild: bool = False) -> dict:
        target = Path(target_path).resolve()
        index_root = ensure_dir(index_dir)
        logger.info(
            "Starting index run: target=%s index_dir=%s rebuild=%s",
            target,
            index_root,
            rebuild,
        )

        if rebuild:
            faiss_dir = index_root / "faiss"
            if faiss_dir.exists():
                logger.debug("Removing existing FAISS directory at %s", faiss_dir)
                shutil.rmtree(faiss_dir)
            manifest = {"files": {}}
            chunk_map: dict[str, CodeChunk] = {}
        else:
            manifest = self._load_manifest(index_root)
            chunk_map = self._load_chunks(index_root)

        current_files = iter_source_files(target)
        logger.info("Discovered %d source files", len(current_files))
        current_hashes = {str(path): sha256_file(path) for path in current_files}
        known_files = set(manifest["files"].keys())
        existing_files = set(current_hashes.keys())

        changed_files = {
            path for path, digest in current_hashes.items() if manifest["files"].get(path, {}).get("sha256") != digest
        }
        deleted_files = known_files - existing_files
        logger.info(
            "Index delta computed: changed=%d deleted=%d unchanged=%d",
            len(changed_files),
            len(deleted_files),
            len(existing_files) - len(changed_files),
        )

        remove_chunk_ids: list[str] = []
        for file_path in sorted(changed_files | deleted_files):
            old = manifest["files"].get(file_path, {})
            remove_chunk_ids.extend(old.get("chunk_ids", []))
            for chunk_id in old.get("chunk_ids", []):
                chunk_map.pop(chunk_id, None)
            manifest["files"].pop(file_path, None)

        new_chunks: list[CodeChunk] = []
        for file_path in sorted(changed_files):
            logger.debug("Parsing and chunking file %s", file_path)
            symbols = self.parser.parse_file(file_path)
            file_chunks = self.chunker.build_chunks(symbols)
            new_chunks.extend(file_chunks)
            for chunk in file_chunks:
                chunk_map[chunk.id] = chunk

            manifest["files"][file_path] = {
                "sha256": current_hashes[file_path],
                "last_indexed_at": datetime.now(timezone.utc).isoformat(),
                "chunk_ids": [chunk.id for chunk in file_chunks],
            }
            logger.debug("Prepared %d chunks for %s", len(file_chunks), file_path)

        write_json(self._manifest_path(index_root), manifest)
        write_jsonl(self._chunks_path(index_root), [chunk.to_dict() for chunk in chunk_map.values()])
        logger.debug(
            "Persisted manifest and chunk catalog: files=%d chunks=%d",
            len(manifest["files"]),
            len(chunk_map),
        )

        self.store.upsert_chunks(index_root, new_chunks, remove_chunk_ids)
        logger.info(
            "Vector store upsert complete: added=%d removed=%d",
            len(new_chunks),
            len(remove_chunk_ids),
        )

        return {
            "indexed_files": len(changed_files),
            "deleted_files": len(deleted_files),
            "total_files": len(existing_files),
            "new_chunks": len(new_chunks),
            "removed_chunks": len(remove_chunk_ids),
            "index_dir": str(index_root),
        }
