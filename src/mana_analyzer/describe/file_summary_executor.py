# mana_analyzer/describe/file_summary_executor.py

import json
import hashlib
from pathlib import Path
from typing import Any, Iterable, Dict, List, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.text_splitter import CharacterTextSplitter


class FileSummaryExecutor:
    def __init__(
        self,
        file_agent: Any | None = None,
        llm_chain: Any | None = None,
        aggregator_chain: Any | None = None,
        *,
        max_source_chars: int = 12_000,
        chunk_size: int = 6_000,
        chunk_overlap: int = 500,
        max_workers: int = 4,
    ) -> None:
        """
        :param file_agent: optional agent with run_batch or invoke/run API
        :param llm_chain: fallback LLMChain with summarize_files_batch(inputs) → List[(str, symbols_dict)]
        :param aggregator_chain: optional LLMChain to aggregate chunk summaries into one file‐level summary
        :param max_source_chars: if file smaller than this, sent as a single chunk
        :param chunk_size: character length of each chunk
        :param chunk_overlap: character overlap between chunks
        :param max_workers: parallelism for summarizing each file
        """
        self.file_agent = file_agent
        self.llm_chain = llm_chain
        self.aggregator_chain = aggregator_chain

        self.max_source_chars = max_source_chars
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_workers = max_workers

    def summarize_files(
        self,
        files: Iterable[Path],
        language_map: Dict[Path, str],
        source_map: Dict[Path, str],
    ) -> Dict[Path, Dict[str, Any]]:
        """
        Summarize each file in parallel. Return:
            { Path → { "summary": str, "symbols": {…} } }
        """
        results: Dict[Path, Dict[str, Any]] = {}

        # If a file_agent is available, try it first
        if self.file_agent:
            agent_inputs = []
            for path in files:
                src = source_map[path]
                # if very large, truncate to max_source_chars
                snippet = src[: self.max_source_chars]
                agent_inputs.append(
                    {"file_path": str(path), "language": language_map.get(path), "source": snippet}
                )

            # try batch
            if hasattr(self.file_agent, "run_batch"):
                try:
                    outputs = self.file_agent.run_batch(agent_inputs)
                except Exception:
                    outputs = [self._safe_run(i) for i in agent_inputs]
            else:
                outputs = [self._safe_run(i) for i in agent_inputs]

            for path, out in zip(files, outputs):
                if not isinstance(out, dict):
                    out = {"summary": str(out), "symbols": {}}
                results[path] = {
                    "summary": out.get("summary", ""),
                    "symbols": out.get("symbols", {}),
                }
            return results

        # Otherwise fall back to LLMChain chunking + optional aggregation
        if not self.llm_chain:
            # no agent, no llm ⇒ blank
            for p in files:
                results[p] = {"summary": "", "symbols": {}}
            return results

        # helper to process a single file
        def _summarize_one(path: Path) -> Tuple[Path, str, Dict[str, Any]]:
            full_src = source_map[path]
            lang = language_map.get(path, "text")

            # decide if we need to chunk
            if len(full_src) <= self.max_source_chars:
                # single‐chunk path
                single_input = [{"file_path": str(path), "language": lang, "source": full_src}]
                summaries = self.llm_chain.summarize_files_batch(single_input)
                summary, symbols = summaries[0]
                return path, summary, symbols

            # else chunk it
            splitter = CharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            chunk_texts = splitter.split_text(full_src)

            # prepare chunk‐level inputs
            chunk_inputs = []
            for idx, txt in enumerate(chunk_texts):
                chunk_inputs.append(
                    {
                        "file_path": f"{path}#chunk{idx}",
                        "language": lang,
                        "source": txt,
                    }
                )

            # call llm_chain in batches of size <= max_workers
            chunk_results: List[Tuple[str, Dict[str, Any]]] = []
            # note llm_chain.summarize_files_batch can accept a list at once
            for i in range(0, len(chunk_inputs), self.max_workers):
                batch = chunk_inputs[i : i + self.max_workers]
                out_batch = self.llm_chain.summarize_files_batch(batch)
                chunk_results.extend(out_batch)

            # chunk_results is List[(summary_str, symbols_dict)]
            # optionally aggregate via aggregator_chain
            if self.aggregator_chain:
                # build JSON array of chunk analyses
                chunk_analyses = [
                    {"summary": s, "symbols": syms} for (s, syms) in chunk_results
                ]
                payload = json.dumps(chunk_analyses, indent=2)
                raw = self.aggregator_chain.run(
                    file_path=str(path), chunk_analyses=payload
                )
                try:
                    parsed = json.loads(raw.strip())
                    return path, parsed.get("summary", ""), parsed.get("symbols", {})
                except Exception:
                    # fallback to simple merge
                    pass

            # fallback local merge: concatenate summaries + union symbols
            merged_summary = "\n".join(s for (s, _) in chunk_results)

            funcs: Set[str] = set()
            classes: Set[str] = set()
            consts: Set[str] = set()
            for (_, syms) in chunk_results:
                for f in syms.get("functions", []):
                    funcs.add(f)
                for c in syms.get("classes", []):
                    classes.add(c)
                for k in syms.get("constants", []):
                    consts.add(k)

            merged_symbols = {
                "functions": sorted(funcs),
                "classes": sorted(classes),
                "constants": sorted(consts),
            }

            return path, merged_summary, merged_symbols

        # run all files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            futures = {exe.submit(_summarize_one, p): p for p in files}
            for fut in as_completed(futures):
                pth, summ, syms = fut.result()
                results[pth] = {"summary": summ, "symbols": syms}

        return results

    def _safe_run(self, inp: dict) -> dict:
        """Invoke the agent safely on a single input dict."""
        try:
            if hasattr(self.file_agent, "invoke"):
                out = self.file_agent.invoke(inp)
            else:
                out = self.file_agent.run(inp)
        except Exception as e:
            out = {"summary": f"[agent error: {e}]", "symbols": {}}

        if not isinstance(out, dict):
            out = {"summary": str(out), "symbols": {}}
        return out
