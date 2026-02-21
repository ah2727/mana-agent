from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path


def setup_logging(verbose: bool = False, log_dir: str | Path | None = None) -> Path:
    level = logging.DEBUG if verbose else logging.INFO
    project_root = Path.cwd().resolve()
    project_name = project_root.name or "project"
    date_tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    root = Path(log_dir).resolve() if log_dir else (project_root / ".mana_logs")
    root.mkdir(parents=True, exist_ok=True)
    log_file = root / f"{date_tag}-{project_name}.log"

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.propagate = False
    return log_file
