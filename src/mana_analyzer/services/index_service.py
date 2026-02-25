from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor

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
        self._executor_workers = max(1, min(8, os.cpu_count() or 4))

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

    def _build_chunks_for_file(self, file_path: str) -> tuple[str, list[CodeChunk]]:
        logger.debug("Gathering chunks for %s via thread pool", file_path)
        try:
            symbols = self.parser.parse_file(file_path)
            file_chunks = self.chunker.build_chunks(symbols)
            return file_path, file_chunks
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.exception("Failed to build chunks for %s: %s", file_path, exc)
            return file_path, []

    def index(
        self,
        target_path: str | Path,
        index_dir: str | Path,
        rebuild: bool = False,
        vectors: bool = True,  # ✅ NEW: allow chunks-only indexing (no embeddings/FAISS)
    ) -> dict:
        target = Path(target_path).resolve()
        index_root = ensure_dir(index_dir)
        logger.info(
            "Starting index run: target=%s index_dir=%s rebuild=%s vectors=%s",
            target,
            index_root,
            rebuild,
            vectors,
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
        if changed_files:
            sorted_changed = sorted(changed_files)
            worker_count = min(len(sorted_changed), self._executor_workers)
            logger.info(
                "Processing %d changed files using %d worker(s)",
                len(sorted_changed),
                worker_count,
            )
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                for file_path, file_chunks in executor.map(self._build_chunks_for_file, sorted_changed):
                    if not file_chunks:
                        logger.warning("No chunks built for %s", file_path)
                        continue
                    new_chunks.extend(file_chunks)
                    for chunk in file_chunks:
                        chunk_map[chunk.id] = chunk
                    manifest["files"][file_path] = {
                        "sha256": current_hashes[file_path],
                        "last_indexed_at": datetime.now(timezone.utc).isoformat(),
                        "chunk_ids": [chunk.id for chunk in file_chunks],
                    }
                    logger.debug("Prepared %d chunks for %s", len(file_chunks), file_path)

        # ✅ Always persist manifest + chunks, even if vectors fail
        write_json(self._manifest_path(index_root), manifest)
        write_jsonl(self._chunks_path(index_root), [chunk.to_dict() for chunk in chunk_map.values()])
        logger.debug(
            "Persisted manifest and chunk catalog: files=%d chunks=%d",
            len(manifest["files"]),
            len(chunk_map),
        )

        wrote_vectors = False
        vector_error = ""

        # ✅ If vectors disabled, skip embeddings entirely
        if not vectors:
            logger.info("Skipping vector upsert (vectors=False). Chunks-only index created.")
        else:
            try:
                self.store.upsert_chunks(index_root, new_chunks, remove_chunk_ids)
                wrote_vectors = True
                logger.info(
                    "Vector store upsert complete: added=%d removed=%d",
                    len(new_chunks),
                    len(remove_chunk_ids),
                )
            except Exception as exc:
                # ✅ Critical: do not crash indexing if embeddings/FAISS fails (e.g., 404)
                vector_error = str(exc)
                wrote_vectors = False
                logger.warning(
                    "Vector store upsert failed; keeping chunks-only index. error=%s",
                    vector_error,
                )

        return {
            "indexed_files": len(changed_files),
            "deleted_files": len(deleted_files),
            "total_files": len(existing_files),
            "new_chunks": len(new_chunks),
            "removed_chunks": len(remove_chunk_ids),
            "index_dir": str(index_root),
            # ✅ NEW fields
            "vectors": wrote_vectors,
            "vector_error": vector_error,
        }