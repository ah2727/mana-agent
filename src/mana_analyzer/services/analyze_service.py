from __future__ import annotations

import logging
from pathlib import Path

from mana_analyzer.analysis.checks import PythonStaticAnalyzer
from mana_analyzer.analysis.models import Finding
from mana_analyzer.utils.io import iter_python_files

logger = logging.getLogger(__name__)


class AnalyzeService:
    def __init__(self, analyzer: PythonStaticAnalyzer) -> None:
        self.analyzer = analyzer

    def analyze(self, target_path: str | Path) -> list[Finding]:
        target = Path(target_path).resolve()
        logger.info("Starting static analysis for %s", target)
        files = iter_python_files(target)
        logger.info("Collected %d files for analysis", len(files))
        findings: list[Finding] = []
        for file_path in files:
            logger.debug("Analyzing file %s", file_path)
            findings.extend(self.analyzer.analyze_file(file_path))
        findings.sort(key=lambda item: (item.file_path, item.line, item.column, item.rule_id))
        logger.info("Static analysis complete: findings=%d", len(findings))
        return findings
