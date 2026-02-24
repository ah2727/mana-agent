from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

from mana_analyzer.analysis.models import Finding
from mana_analyzer.llm.prompts import ANALYZE_HUMAN_TEMPLATE, ANALYZE_SYSTEM_PROMPT
from mana_analyzer.llm.run_logger import LlmRunLogger

logger = logging.getLogger(__name__)


class AnalyzeChain:
    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        logger.debug("Initializing analyze chain with model=%s", model)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ANALYZE_SYSTEM_PROMPT),
                ("human", ANALYZE_HUMAN_TEMPLATE),
            ]
        )
        kwargs: dict[str, Any] = {"api_key": api_key, "model": model}
        if base_url:
            kwargs["base_url"] = base_url
        self.llm = ChatOpenAI(**kwargs)
        self.model = model
        self.run_logger = LlmRunLogger()

    @staticmethod
    def _normalize_finding(item: Any, fallback_file_path: str) -> Finding | None:
        if not isinstance(item, dict):
            return None

        message = str(item.get("message", "")).strip()
        if not message:
            return None

        raw_rule_id = str(item.get("rule_id", "")).strip() or "llm-generic"
        rule_id = raw_rule_id if raw_rule_id.startswith("llm-") else f"llm-{raw_rule_id}"

        severity = str(item.get("severity", "warning")).strip().lower()
        if severity not in {"warning", "error"}:
            severity = "warning"

        file_path = str(item.get("file_path", "")).strip() or fallback_file_path

        try:
            line = int(item.get("line", 1))
        except (TypeError, ValueError):
            line = 1
        if line < 1:
            line = 1

        try:
            column = int(item.get("column", 0))
        except (TypeError, ValueError):
            column = 0
        if column < 0:
            column = 0

        return Finding(
            rule_id=rule_id,
            severity=severity,
            message=message,
            file_path=file_path,
            line=line,
            column=column,
        )

    def run(self, file_path: str, source: str, static_findings: list[Finding]) -> list[Finding]:
        logger.info("Invoking LLM analyzer for %s", file_path)
        chain = self.prompt | self.llm
        started = perf_counter()
        response = chain.invoke(
            {
                "file_path": file_path,
                "source": source,
                "static_findings": json.dumps([item.to_dict() for item in static_findings]),
            }
        )
        elapsed_ms = (perf_counter() - started) * 1000
        text = str(response.content).strip()
        findings: list[Finding] = []
        parse_status = "ok"

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("LLM analyze output is not valid JSON for %s", file_path)
            parsed = []
            parse_status = "invalid-json"

        if not isinstance(parsed, list):
            logger.warning("LLM analyze output is not a list for %s", file_path)
            parsed = []
            parse_status = "not-list"

        dropped = 0
        for item in parsed:
            finding = self._normalize_finding(item, fallback_file_path=file_path)
            if finding is None:
                dropped += 1
                continue
            findings.append(finding)

        if dropped:
            logger.warning("Dropped %d malformed LLM findings for %s", dropped, file_path)
        run_logger = getattr(self, "run_logger", None)
        if run_logger is not None:
            run_logger.log(
                {
                    "flow": "analyze",
                    "model": getattr(self, "model", "unknown"),
                    "file_path": file_path,
                    "source_chars": len(source),
                    "source": source,
                    "static_findings_count": len(static_findings),
                    "static_findings": [item.to_dict() for item in static_findings],
                    "result_findings_count": len(findings),
                    "result_findings": [item.to_dict() for item in findings],
                    "parse_status": parse_status,
                    "duration_ms": round(elapsed_ms, 3),
                    "response": text,
                }
            )
        logger.info("LLM analyzer produced %d findings for %s", len(findings), file_path)
        return findings
