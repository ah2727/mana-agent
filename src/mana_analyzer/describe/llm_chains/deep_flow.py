import os
import json
import time
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DeepFlowChain:
    """
    A LangChain-powered deep analysis chain that:
      - discovers & reads files
      - splits them into chunks
      - summarizes chunks in parallel (with caching)
      - aggregates chunk summaries per file
      - runs a final deep-flow analysis on all file summaries
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        openai_api_key: str,
        base_url: Optional[str] = None,
        max_retries: int = 999999999999,
        retry_delay: int = 5,
    ) -> None:
        # --- LLM client & retry config ---
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            temperature=temperature,
            base_url=base_url,
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.output_parser = StrOutputParser()

        # --- Deep flow final analysis chain ---
        deep_template = """You are an expert code analysis assistant performing a deep security and architecture review.

Security Lens: {security_lens}
Target Length: ~{line_target} lines of markdown

## Dependency Report
{dependency_report}

## Structure Summary
{structure_summary}

## Findings Summary
{findings_summary}

## Security Summary
{security_summary}

## Sampled File Summaries
{sampled_file_summaries}

Based on the above, produce a cohesive, in-depth analysis covering:
- System architecture and key execution flows
- Trust boundaries and attack surface
- Potential edge cases, vulnerabilities, or issues
- Hardening priorities and suggestions for improvement

Return your output as markdown. Be defensive-only; never include exploit code or procedures."""
        self.deep_prompt = PromptTemplate(
            input_variables=[
                "security_lens",
                "line_target",
                "dependency_report",
                "structure_summary",
                "findings_summary",
                "security_summary",
                "sampled_file_summaries",
            ],
            template=deep_template,
        )
        self.deep_chain = self.deep_prompt | self.llm | self.output_parser

        # --- Chunk summarization chain (per‐chunk) ---
        summary_template = """You are a code analysis expert. Analyze the following source code snippet (a chunk of a file) and provide:
1. A concise summary of what this snippet does (2-3 sentences)
2. Key symbols (functions, classes, constants) defined or referenced in this snippet

File Chunk Path: {chunk_path}
Language: {language}
Source Code:
{source}

Respond in JSON format:
{{
  "summary": "Brief description of the snippet's functionality",
  "symbols": {{
    "functions": ["function1", "function2"],
    "classes": ["Class1", "Class2"],
    "constants": ["CONST1", "CONST2"]
  }}
}}"""
        self.summary_prompt = PromptTemplate(
            input_variables=["chunk_path", "language", "source"], template=summary_template
        )
        self.summary_chain = self.summary_prompt | self.llm | self.output_parser

        # --- Aggregator chain (per‐file) ---
        aggregator_template = """You are a code analysis expert. Given the following JSON array of chunk-level analyses for file {file_path}, produce:
1) A consolidated file-level summary (2-3 sentences)
2) A combined list of unique key symbols (functions, classes, constants)

Chunk Analyses:
{chunk_analyses}

Respond in JSON format:
{{
  "summary": "Brief description of the file's overall functionality",
  "symbols": {{
    "functions": [...],
    "classes": [...],
    "constants": [...]
  }}
}}"""
        self.aggregator_prompt = PromptTemplate(
            input_variables=["file_path", "chunk_analyses"], template=aggregator_template
        )
        self.aggregator_chain = self.aggregator_prompt | self.llm | self.output_parser

        # --- JSON repair chain ---
        repair_template = """The following text was supposed to be valid JSON but it is not.
Extract or fix it and return ONLY valid JSON matching this schema:
{{
  "summary": "string",
  "symbols": {{
    "functions": [],
    "classes": [],
    "constants": []
  }}
}}

Broken text:
{broken_text}

Return ONLY the corrected JSON, no explanation, no markdown fences."""
        self.repair_prompt = PromptTemplate(
            input_variables=["broken_text"], template=repair_template
        )
        self.repair_chain = self.repair_prompt | self.llm | self.output_parser

        # --- In-memory cache: { file_hash: { chunk_index: (summary, symbols) } } ---
        self.chunk_cache: Dict[str, Dict[int, Tuple[str, Dict[str, Any]]]] = {}

    def _run_with_retry(self, chain, **kwargs) -> str:
        """Execute chain.invoke with retry on transient errors."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return chain.invoke(kwargs)
            except Exception as e:
                last_error = e
                msg = str(e).lower()
                retryable = any(k in msg for k in ["timeout", "500", "504", "503", "502", "connection", "unavailable"])
                if not retryable or attempt == self.max_retries - 1:
                    raise
                wait = self.retry_delay
                print(f"[DeepFlowChain] API error (attempt {attempt+1}/{self.max_retries}), retrying in {wait}s...")
                time.sleep(wait)
        raise last_error

    def _extract_json(self, raw: str) -> Optional[Dict[str, Any]]:
        """
        Try to parse JSON from raw LLM output.
        Handles: plain JSON, markdown-fenced JSON, JSON buried in text.
        Returns parsed dict or None.
        """
        text = raw.strip()
        if not text:
            return None

        # 1) Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2) Extract from markdown code fence
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 3) Find first { ... last }
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            try:
                return json.loads(text[first_brace : last_brace + 1])
            except json.JSONDecodeError:
                pass

        return None

    def _repair_json_via_llm(self, broken: str) -> Dict[str, Any]:
        """
        Ask the LLM to fix a broken JSON response.
        """
        repaired_raw = self._run_with_retry(self.repair_chain, broken_text=broken[:3000])
        parsed = self._extract_json(repaired_raw)
        if parsed is not None:
            return parsed

        # Last resort fallback
        return {"summary": "", "symbols": {"functions": [], "classes": [], "constants": []}}

    def discover_files(
        self,
        root_path: str,
        include_patterns: List[str],
        exclude_patterns: Optional[List[str]] = None,
    ) -> List[Path]:
        """Walk root_path and return files matching include_patterns minus exclude_patterns."""
        root = Path(root_path)
        paths = set()
        for pat in include_patterns:
            for p in root.rglob(pat):
                if exclude_patterns and any(p.match(exc) for exc in exclude_patterns):
                    continue
                if p.is_file():
                    paths.add(p)
        return sorted(paths)

    def _hash_file(self, path: Path) -> str:
        data = path.read_bytes()
        return hashlib.sha256(data).hexdigest()

    def _split_file_into_chunks(
        self, path: Path, chunk_size: int, chunk_overlap: int
    ) -> List[Document]:
        """Load a file and split into Document chunks with metadata."""
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(docs)
        # annotate metadata
        for idx, doc in enumerate(chunks):
            doc.metadata["path"] = str(path)
            doc.metadata["chunk_index"] = idx
        return chunks

    def _summarize_chunk(
        self, doc: Document, language: str
    ) -> Tuple[int, str, Dict[str, Any]]:
        """
        Summarize a single chunk, with in-memory cache.
        Returns (chunk_index, summary_text, symbols_dict).
        """
        file_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
        idx = doc.metadata["chunk_index"]
        # check cache
        if file_hash in self.chunk_cache and idx in self.chunk_cache[file_hash]:
            return idx, *self.chunk_cache[file_hash][idx]

        # run LLM
        raw = self._run_with_retry(
            self.summary_chain,
            chunk_path=f"{doc.metadata['path']}#chunk{idx}",
            language=language,
            source=doc.page_content,
        )

        # Robust JSON extraction
        parsed = self._extract_json(raw)
        if parsed is None:
            print(f"[DeepFlowChain] Invalid JSON from LLM for chunk {idx}, attempting repair...")
            parsed = self._repair_json_via_llm(raw)

        summary = parsed.get("summary", "")
        symbols = parsed.get("symbols", {})
        # store
        self.chunk_cache.setdefault(file_hash, {})[idx] = (summary, symbols)
        return idx, summary, symbols

    def _aggregate_file_summary(
        self, file_path: Path, chunk_results: List[Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Given chunk-level results for a file, produce a single file summary+symbols.
        chunk_results: list of {"summary":…, "symbols":{…}} entries
        """
        payload = json.dumps(chunk_results, indent=2)
        raw = self._run_with_retry(
            self.aggregator_chain,
            file_path=str(file_path),
            chunk_analyses=payload,
        )

        parsed = self._extract_json(raw)
        if parsed is None:
            print(f"[DeepFlowChain] Invalid JSON from aggregator for {file_path}, attempting repair...")
            parsed = self._repair_json_via_llm(raw)

        return parsed.get("summary", ""), parsed.get("symbols", {})

    def analyze_directory(
        self,
        root_path: str,
        include_patterns: List[str] = ["**/*.py"],
        exclude_patterns: Optional[List[str]] = None,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        max_workers: int = 4,
        # pass-through to synthesize_deep_flow_analysis:
        dependency_report: Any = None,
        structure_summary: Any = None,
        findings_summary: Any = None,
        security_summary: Any = None,
        line_target: int = 350,
        security_lens: str = "defensive-red-team",
    ) -> str:
        """
        Discover, chunk, summarize, aggregate, then run final deep analysis.
        Returns the final markdown analysis.
        """
        files = self.discover_files(root_path, include_patterns, exclude_patterns)
        file_summaries: List[Dict[str, Any]] = []

        for path in files:
            language = path.suffix.lstrip(".") or "text"
            chunks = self._split_file_into_chunks(path, chunk_size, chunk_overlap)

            # summarize chunks in parallel
            chunk_results: Dict[int, Dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                futures = {
                    exe.submit(self._summarize_chunk, doc, language): doc
                    for doc in chunks
                }
                for fut in as_completed(futures):
                    idx, summ, syms = fut.result()
                    chunk_results[idx] = {"summary": summ, "symbols": syms}

            # order by chunk index
            ordered = [chunk_results[i] for i in sorted(chunk_results.keys())]
            file_summary, file_symbols = self._aggregate_file_summary(path, ordered)
            file_summaries.append(
                {
                    "path": str(path),
                    "summary": file_summary,
                    "symbols": file_symbols,
                }
            )

        # final deep analysis
        sampled_file_summaries = json.dumps(file_summaries, indent=2)
        return self.synthesize_deep_flow_analysis(
            dependency_report=dependency_report,
            structure_summary=structure_summary,
            findings_summary=findings_summary,
            security_summary=security_summary,
            sampled_file_summaries=sampled_file_summaries,
            line_target=line_target,
            security_lens=security_lens,
        )

    def synthesize_deep_flow_analysis(
        self,
        *,
        dependency_report: Any = None,
        structure_summary: Any = None,
        findings_summary: Any = None,
        security_summary: Any = None,
        sampled_file_summaries: Any = None,
        line_target: int = 350,
        security_lens: str = "defensive-red-team",
        flows: Any = None,
        detail_line_target: Optional[int] = None,
        **kwargs,
    ) -> str:
        effective_line_target = detail_line_target if detail_line_target is not None else line_target

        def _serialize(obj: Any) -> str:
            if obj is None:
                return "Not available."
            if isinstance(obj, str):
                return obj
            if hasattr(obj, "to_dict"):
                return json.dumps(obj.to_dict(), indent=2, default=str)
            try:
                return json.dumps(obj, indent=2, default=str)
            except (TypeError, ValueError):
                return str(obj)

        output = self._run_with_retry(
            self.deep_chain,
            security_lens=security_lens,
            line_target=effective_line_target,
            dependency_report=_serialize(dependency_report),
            structure_summary=_serialize(structure_summary),
            findings_summary=_serialize(findings_summary),
            security_summary=_serialize(security_summary),
            sampled_file_summaries=_serialize(sampled_file_summaries),
        )
        return output.strip()

    def summarize_files_batch(
        self,
        batch: List[Dict[str, Any]]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Implements the interface FileSummaryExecutor expects:
        Input: a list of dicts { file_path, language, source }
        Output: a list of (summary_str, symbols_dict)
        """
        results: List[Tuple[str, Dict[str, Any]]] = []
        for item in batch:
            src = item["source"]
            path_str = item["file_path"]
            lang = item.get("language", "text")

            # Build a Document with the same metadata keys
            doc = Document(
                page_content=src,
                metadata={
                    "path": path_str,
                    "chunk_index": 0
                },
            )

            # _summarize_chunk returns (idx, summary, symbols)
            idx, summary, symbols = self._summarize_chunk(doc, language=lang)
            results.append((summary, symbols))
        return results
