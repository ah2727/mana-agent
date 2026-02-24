from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from mana_analyzer.analysis.checks import PythonStaticAnalyzer
from mana_analyzer.analysis.models import Finding
from mana_analyzer.utils.io import iter_python_files

logger = logging.getLogger(__name__)


def _analyze_file_worker(path: str) -> list[Finding]:
    analyzer = PythonStaticAnalyzer()
    return analyzer.analyze_file(path)


class AnalyzeService:
    def __init__(self, analyzer: PythonStaticAnalyzer) -> None:
        self.analyzer = analyzer
        self._worker_count = max(1, os.cpu_count() or 2)

    def analyze(self, target_path: str | Path) -> list[Finding]:
        target = Path(target_path).resolve()
        logger.info("Starting static analysis for %s", target)
        files = iter_python_files(target)
        logger.info("Collected %d files for analysis", len(files))
        findings: list[Finding] = []
        worker_count = min(len(files), self._worker_count)
        logger.info("Analyzing files using %d worker(s)", worker_count or 1)
        if worker_count <= 1:
            for file_path in files:
                logger.debug("Analyzing file %s", file_path)
                findings.extend(self.analyzer.analyze_file(file_path))
        else:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_to_path = {
                    executor.submit(_analyze_file_worker, str(file_path)): str(file_path)
                    for file_path in files
                }
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        results = future.result()
                    except Exception as exc:
                        logger.warning(
                            "Process-level analysis failed for %s: %s",
                            path,
                            exc,
                        )
                        continue
                    findings.extend(results)
        findings.sort(key=lambda item: (item.file_path, item.line, item.column, item.rule_id))
        logger.info("Static analysis complete: findings=%d", len(findings))
        return findings
