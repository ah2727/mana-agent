from __future__ import annotations

import logging
from pathlib import Path

from mana_analyzer.analysis.models import AskResponse, AskResponseWithTrace, SearchHit, SourceGroup
from mana_analyzer.llm.ask_agent import AskAgent
from mana_analyzer.llm.qna_chain import QnAChain
from mana_analyzer.services.search_service import SearchService
from mana_analyzer.vector_store.faiss_store import FaissStore

logger = logging.getLogger(__name__)


class AskService:
    def __init__(
        self,
        store: FaissStore,
        qna_chain: QnAChain,
        ask_agent: AskAgent | None = None,
        search_service: SearchService | None = None,
    ) -> None:
        self.store = store
        self.qna_chain = qna_chain
        self.ask_agent = ask_agent
        self.search_service = search_service

    @staticmethod
    def _render_context(sources: list[SearchHit]) -> str:
        blocks: list[str] = []
        for src in sources:
            blocks.append(
                "\n".join(
                    [
                        f"source: {src.file_path}:{src.start_line}-{src.end_line}",
                        f"symbol: {src.symbol_name}",
                        "snippet:",
                        src.snippet,
                    ]
                )
            )
        return "\n\n---\n\n".join(blocks)

    def ask(self, index_dir: str | Path, question: str, k: int) -> AskResponse:
        resolved_index = Path(index_dir).resolve()
        logger.info("Running ask flow: index_dir=%s k=%d", resolved_index, k)
        logger.debug("Question: %s", question)
        sources = self.store.search(resolved_index, query=question, k=k)
        logger.info("Retrieved context sources: %d", len(sources))
        if not sources:
            message = (
                "I could not find relevant indexed code context. "
                "Re-run indexing or provide a narrower question."
            )
            logger.warning("No sources found for ask flow")
            return AskResponse(answer=message, sources=[])

        context = self._render_context(sources)
        logger.debug("Rendered context length: %d chars", len(context))
        answer = self.qna_chain.run(question=question, context=context)
        logger.info("LLM answer generated")
        return AskResponse(answer=answer, sources=sources)

    def ask_with_tools(
        self,
        index_dir: str | Path,
        question: str,
        k: int,
        max_steps: int = 6,
        timeout_seconds: int = 30,
    ) -> AskResponseWithTrace:
        if self.ask_agent is None:
            raise RuntimeError("ask agent is not configured")
        try:
            return self.ask_agent.run(
                question=question,
                index_dir=index_dir,
                k=k,
                max_steps=max_steps,
                timeout_seconds=timeout_seconds,
            )
        except Exception:
            logger.exception("ask agent failed; falling back to classic ask flow")
            fallback = self.ask(index_dir=index_dir, question=question, k=k)
            return AskResponseWithTrace(
                answer=fallback.answer,
                sources=fallback.sources,
                mode="classic-fallback",
                trace=[],
                warnings=[],
            )

    @staticmethod
    def _group_sources_by_index(sources: list[SearchHit], index_dirs: list[Path]) -> list[SourceGroup]:
        grouped: dict[Path, list[SearchHit]] = {item.resolve(): [] for item in index_dirs}
        for source in sources:
            source_path = Path(source.file_path).resolve()
            matched: Path | None = None
            for index_dir in grouped.keys():
                subproject_root = index_dir.parent
                if source_path == subproject_root or subproject_root in source_path.parents:
                    if matched is None or len(str(subproject_root)) > len(str(matched.parent)):
                        matched = index_dir
            if matched is not None:
                grouped[matched].append(source)

        payload: list[SourceGroup] = []
        for index_dir in sorted(grouped.keys(), key=lambda item: str(item)):
            hits = grouped[index_dir]
            if not hits:
                continue
            payload.append(
                SourceGroup(
                    index_dir=str(index_dir),
                    subproject_root=str(index_dir.parent),
                    sources=hits,
                )
            )
        return payload

    def ask_dir_mode(
        self,
        index_dirs: list[str | Path],
        question: str,
        k: int,
        root_dir: str | Path,
    ) -> AskResponse:
        if self.search_service is None:
            raise RuntimeError("search service is not configured")
        resolved_indexes = sorted({Path(item).resolve() for item in index_dirs}, key=lambda item: str(item))
        if not resolved_indexes:
            root = Path(root_dir).resolve()
            answer = (
                f"No usable indexes found under {root}. "
                f"Run: mana-analyzer index {root} or re-run ask with --auto-index-missing."
            )
            return AskResponse(answer=answer, sources=[], source_groups=[], warnings=[answer])

        sources, warnings = self.search_service.search_multi(index_dirs=resolved_indexes, query=question, k=k)
        if not sources:
            answer = (
                "I could not find relevant indexed code context across discovered indexes. "
                "Try a narrower question or rebuild indexes."
            )
            return AskResponse(answer=answer, sources=[], source_groups=[], warnings=warnings)

        context = self._render_context(sources)
        answer = self.qna_chain.run(question=question, context=context)
        return AskResponse(
            answer=answer,
            sources=sources,
            source_groups=self._group_sources_by_index(sources, resolved_indexes),
            warnings=warnings,
        )

    def ask_with_tools_dir_mode(
        self,
        index_dirs: list[str | Path],
        question: str,
        k: int,
        max_steps: int = 6,
        timeout_seconds: int = 30,
        root_dir: str | Path | None = None,
    ) -> AskResponseWithTrace:
        if self.ask_agent is None:
            raise RuntimeError("ask agent is not configured")
        resolved_indexes = sorted({Path(item).resolve() for item in index_dirs}, key=lambda item: str(item))
        if not resolved_indexes:
            root = Path(root_dir or Path.cwd()).resolve()
            answer = (
                f"No usable indexes found under {root}. "
                f"Run: mana-analyzer index {root} or re-run ask with --auto-index-missing."
            )
            return AskResponseWithTrace(
                answer=answer,
                sources=[],
                source_groups=[],
                warnings=[answer],
                mode="agent-tools",
                trace=[],
            )
        result = self.ask_agent.run_multi(
            question=question,
            index_dirs=resolved_indexes,
            k=k,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
        )
        result.source_groups = self._group_sources_by_index(result.sources, resolved_indexes)
        return result
