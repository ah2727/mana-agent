from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

from mana_analyzer.llm.run_logger import LlmRunLogger

logger = logging.getLogger(__name__)

FILE_SUMMARY_SYSTEM = """
You are a code summarizer.
Return strict JSON with keys: summary (string), symbols (array of strings).
Keep summary concise and factual.
""".strip()

FILE_SUMMARY_HUMAN = """
File path: {file_path}
Language: {language}
Source:
{source}
""".strip()

ARCH_SYSTEM = """
You are a software architecture analyst.
Return strict JSON with keys: architecture_summary (string), tech_summary (string).
Use only the provided dependency and file-summary data.
""".strip()

ARCH_HUMAN = """
Dependency report JSON:
{dependency_report}

File summaries JSON:
{file_summaries}
""".strip()

TECH_SYSTEM = """
You are a repository technology detector.
Return strict JSON with key frameworks as an array of short framework names.
""".strip()

TECH_HUMAN = """
Project sample files:
{samples}
""".strip()


class RepositoryMultiChain:
    _MAX_FRAMEWORKS = 24
    _MAX_DEPENDENCIES = 200
    _MAX_EDGES = 400
    _MAX_FILE_SUMMARIES = 20
    _MAX_SYMBOLS_PER_FILE = 24
    _MAX_SUMMARY_CHARS = 700

    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        kwargs: dict[str, Any] = {"api_key": api_key, "model": model}
        if base_url:
            kwargs["base_url"] = base_url
        self.llm = ChatOpenAI(**kwargs)
        self.run_logger = LlmRunLogger()
        self.model = model

        self.file_summary_prompt = ChatPromptTemplate.from_messages(
            [("system", FILE_SUMMARY_SYSTEM), ("human", FILE_SUMMARY_HUMAN)]
        )
        self.arch_prompt = ChatPromptTemplate.from_messages([( "system", ARCH_SYSTEM), ("human", ARCH_HUMAN)])
        self.tech_prompt = ChatPromptTemplate.from_messages([( "system", TECH_SYSTEM), ("human", TECH_HUMAN)])

    @staticmethod
    def _safe_json(text: str) -> dict[str, Any]:
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
        return {}

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        if max_chars < 4:
            return text[:max_chars]
        return text[: max_chars - 3].rstrip() + "..."

    @classmethod
    def _compact_dependency_report(cls, dependency_report: dict[str, Any]) -> dict[str, Any]:
        compact = {
            "project_root": dependency_report.get("project_root"),
            "package_managers": list(dependency_report.get("package_managers", []))[: cls._MAX_FRAMEWORKS],
            "frameworks": list(dependency_report.get("frameworks", []))[: cls._MAX_FRAMEWORKS],
            "technologies": list(dependency_report.get("technologies", []))[: cls._MAX_FRAMEWORKS],
            "runtime_dependencies": list(dependency_report.get("runtime_dependencies", []))[: cls._MAX_DEPENDENCIES],
            "dev_dependencies": list(dependency_report.get("dev_dependencies", []))[: cls._MAX_DEPENDENCIES],
            "manifests": list(dependency_report.get("manifests", []))[: cls._MAX_DEPENDENCIES],
            "languages": list(dependency_report.get("languages", []))[: cls._MAX_FRAMEWORKS],
        }
        module_edges = list(dependency_report.get("module_edges", []))[: cls._MAX_EDGES]
        dependency_edges = list(dependency_report.get("dependency_edges", []))[: cls._MAX_EDGES]
        compact["module_edges"] = [
            {
                "source": edge.get("source"),
                "target": edge.get("target"),
                "kind": edge.get("kind"),
            }
            for edge in module_edges
            if isinstance(edge, dict)
        ]
        compact["dependency_edges"] = [
            {
                "source": edge.get("source"),
                "target": edge.get("target"),
                "kind": edge.get("kind"),
            }
            for edge in dependency_edges
            if isinstance(edge, dict)
        ]
        return compact

    @classmethod
    def _compact_file_summaries(cls, file_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for item in file_summaries[: cls._MAX_FILE_SUMMARIES]:
            if not isinstance(item, dict):
                continue
            symbols = [str(value) for value in item.get("symbols", []) if str(value).strip()]
            compact.append(
                {
                    "file_path": str(item.get("file_path", "")),
                    "language": str(item.get("language", "")),
                    "symbols": symbols[: cls._MAX_SYMBOLS_PER_FILE],
                    "summary": cls._truncate(str(item.get("summary", "")), cls._MAX_SUMMARY_CHARS),
                }
            )
        return compact

    def summarize_file(self, file_path: Path, language: str, source: str) -> tuple[str, list[str]]:
        chain = self.file_summary_prompt | self.llm
        started = perf_counter()
        response = chain.invoke({"file_path": str(file_path), "language": language, "source": source})
        elapsed_ms = (perf_counter() - started) * 1000
        payload = self._safe_json(str(response.content))
        summary = str(payload.get("summary", "")).strip() or "No summary generated."
        symbols = [str(item) for item in payload.get("symbols", []) if str(item).strip()]
        self.run_logger.log(
            {
                "flow": "repo-file-summary",
                "model": self.model,
                "file_path": str(file_path),
                "source_chars": len(source),
                "duration_ms": round(elapsed_ms, 3),
                "response": str(response.content),
            }
        )
        return summary, symbols

    def synthesize_architecture(self, dependency_report: dict[str, Any], file_summaries: list[dict[str, Any]]) -> tuple[str, str]:
        compact_dependency_report = self._compact_dependency_report(dependency_report)
        compact_file_summaries = self._compact_file_summaries(file_summaries)
        chain = self.arch_prompt | self.llm
        started = perf_counter()
        response = chain.invoke(
            {
                "dependency_report": json.dumps(compact_dependency_report, ensure_ascii=False),
                "file_summaries": json.dumps(compact_file_summaries, ensure_ascii=False),
            }
        )
        elapsed_ms = (perf_counter() - started) * 1000
        payload = self._safe_json(str(response.content))
        architecture = str(payload.get("architecture_summary", "")).strip() or "Architecture summary unavailable."
        tech = str(payload.get("tech_summary", "")).strip() or "Technology summary unavailable."
        self.run_logger.log(
            {
                "flow": "repo-architecture-summary",
                "model": self.model,
                "summary_count": len(compact_file_summaries),
                "duration_ms": round(elapsed_ms, 3),
                "response": str(response.content),
            }
        )
        return architecture, tech

    def detect_frameworks_from_samples(self, samples: list[dict[str, str]]) -> list[str]:
        if not samples:
            return []
        chain = self.tech_prompt | self.llm
        started = perf_counter()
        response = chain.invoke({"samples": json.dumps(samples, ensure_ascii=False)})
        elapsed_ms = (perf_counter() - started) * 1000
        payload = self._safe_json(str(response.content))
        frameworks = sorted({str(item).strip() for item in payload.get("frameworks", []) if str(item).strip()})
        self.run_logger.log(
            {
                "flow": "repo-tech-detection",
                "model": self.model,
                "sample_count": len(samples),
                "duration_ms": round(elapsed_ms, 3),
                "response": str(response.content),
            }
        )
        return frameworks
