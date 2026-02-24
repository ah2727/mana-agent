from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain.vectorstores import FAISS

from mana_analyzer.analysis.models import CodeChunk, SearchHit
from mana_analyzer.utils.io import ensure_dir

logger = logging.getLogger(__name__)


class FaissStore:
    def __init__(self, embeddings: Any) -> None:
        self.embeddings = embeddings

    @staticmethod
    def _faiss_dir(index_dir: Path) -> Path:
        return index_dir / "faiss"

    def load(self, index_dir: Path) -> FAISS | None:
        faiss_dir = self._faiss_dir(index_dir)
        if not faiss_dir.exists():
            logger.debug("FAISS directory missing at %s", faiss_dir)
            return None
        logger.debug("Loading FAISS index from %s", faiss_dir)
        return FAISS.load_local(
            str(faiss_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def save(self, store: FAISS, index_dir: Path) -> None:
        faiss_dir = self._faiss_dir(index_dir)
        ensure_dir(faiss_dir)
        logger.debug("Saving FAISS index to %s", faiss_dir)
        store.save_local(str(faiss_dir))

    def upsert_chunks(self, index_dir: Path, chunks: list[CodeChunk], delete_ids: list[str]) -> None:
        index_dir = ensure_dir(index_dir)
        logger.info(
            "Upserting vectors: index_dir=%s add=%d delete=%d",
            index_dir,
            len(chunks),
            len(delete_ids),
        )
        store = self.load(index_dir)

        if store is not None and delete_ids:
            logger.debug("Deleting %d vectors from existing FAISS index", len(delete_ids))
            store.delete(ids=delete_ids)

        if chunks:
            texts = [chunk.text for chunk in chunks]
            metadatas = [
                {
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "symbol_name": chunk.symbol_name,
                    "symbol_kind": chunk.symbol_kind,
                    "chunk_id": chunk.id,
                }
                for chunk in chunks
            ]
            ids = [chunk.id for chunk in chunks]
            logger.debug("Embedding %d documents", len(texts))
            vectors = self.embeddings.embed_documents(texts)
            pairs = list(zip(texts, vectors, strict=False))

            if store is None:
                logger.debug("Creating new FAISS index")
                store = FAISS.from_embeddings(
                    text_embeddings=pairs,
                    embedding=self.embeddings,
                    metadatas=metadatas,
                    ids=ids,
                )
            else:
                logger.debug("Appending embeddings to existing FAISS index")
                store.add_embeddings(
                    text_embeddings=pairs,
                    metadatas=metadatas,
                    ids=ids,
                )

        if store is not None:
            self.save(store, index_dir)
            logger.info("FAISS index persisted")
        else:
            logger.info("No FAISS changes to persist")

    def search(self, index_dir: Path, query: str, k: int) -> list[SearchHit]:
        logger.info("Searching FAISS index: index_dir=%s k=%d", index_dir, k)
        logger.debug("Vector search query: %s", query)
        store = self.load(index_dir)
        if store is None:
            logger.warning("Cannot search: FAISS index not found in %s", index_dir)
            return []

        docs = store.similarity_search_with_relevance_scores(query, k=k)
        logger.debug("Raw vector search returned %d docs", len(docs))
        hits: list[SearchHit] = []
        for document, score in docs:
            meta = document.metadata or {}
            hits.append(
                SearchHit(
                    score=float(score),
                    file_path=str(meta.get("file_path", "")),
                    start_line=int(meta.get("start_line", 1)),
                    end_line=int(meta.get("end_line", 1)),
                    symbol_name=str(meta.get("symbol_name", "unknown")),
                    snippet=document.page_content[:500],
                )
            )
        logger.info("Returning %d search hits", len(hits))
        return hits
