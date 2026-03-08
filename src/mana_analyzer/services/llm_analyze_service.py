from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

from mana_analyzer.analysis.models import Finding
from mana_analyzer.llm.analyze_chain import AnalyzeChain
from mana_analyzer.utils.io import iter_python_files

logger = logging.getLogger(__name__)


class LlmAnalyzeService:
    def __init__(self, analyze_chain: AnalyzeChain) -> None:
        self.analyze_chain = analyze_chain

    @staticmethod
    def _select_files(target_path: Path, static_findings: list[Finding], max_files: int) -> list[Path]:
        files = iter_python_files(target_path)
        if max_files < 1:
            return []
        if target_path.is_file():
            return files[:1]
        if not files:
            return []

        counts: dict[str, int] = {}
        for finding in static_findings:
            counts[finding.file_path] = counts.get(finding.file_path, 0) + 1

        if counts:
            ranked = sorted(files, key=lambda item: (-counts.get(str(item), 0), str(item)))
        else:
            ranked = sorted(files, key=str)
        return ranked[:max_files]
    
    def _replace_quotes_with_backticks(text):
        if not isinstance(text, str):
            return text
        return text.replace('"', '`')

    @staticmethod
    def _dedupe_findings(findings: list[Finding]) -> list[Finding]:
        seen: set[tuple[str, str, str, int, int, str]] = set()
        deduped: list[Finding] = []
        for finding in findings:
            key = (
                finding.rule_id,
                finding.severity,
                finding.file_path,
                finding.line,
                finding.column,
                finding.message,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(finding)
        return deduped

    def analyze(self, target_path: str | Path, static_findings: list[Finding], max_files: int = 10) -> list[Finding]:
        target = Path(target_path).resolve()
        selected_files = self._select_files(target, static_findings, max_files)
        logger.info(
            "LLM analyze selected %d files (max=%d) from target=%s",
            len(selected_files),
            max_files,
            target,
        )
        if not selected_files:
            return []

        findings_by_file: dict[str, list[Finding]] = {}
        for finding in static_findings:
            findings_by_file.setdefault(finding.file_path, []).append(finding)

        all_findings: list[Finding] = []
        start = perf_counter()
        for file_path in selected_files:
            file_start = perf_counter()
            try:
                source = file_path.read_text(encoding="utf-8").replace('"', '\\"')
            except OSError:
                logger.warning("Failed reading file for LLM analysis: %s", file_path)
                continue

            static_for_file = findings_by_file.get(str(file_path), [])
            logger.debug(
                "Running LLM analyze for %s with %d static hints",
                file_path,
                len(static_for_file),
            )
            file_findings = self.analyze_chain.run(
                file_path=str(file_path),
                source=source,
                static_findings=static_for_file,
            )
            all_findings.extend(file_findings)
            elapsed_ms = (perf_counter() - file_start) * 1000
            logger.info(
                "LLM analyze completed for %s in %.2fms with %d findings",
                file_path,
                elapsed_ms,
                len(file_findings),
            )

        deduped = self._dedupe_findings(all_findings)
        deduped.sort(key=lambda item: (item.file_path, item.line, item.column, item.rule_id))
        total_ms = (perf_counter() - start) * 1000
        logger.info(
            "LLM analyze run complete in %.2fms: raw=%d deduped=%d",
            total_ms,
            len(all_findings),
            len(deduped),
        )
        return deduped
