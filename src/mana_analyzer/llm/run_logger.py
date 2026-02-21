from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from mana_analyzer.utils.io import ensure_dir


class LlmRunLogger:
    def __init__(self, log_file: str | Path | None = None) -> None:
        env_path = os.getenv("MANA_LLM_LOG_FILE")
        if log_file:
            resolved = Path(log_file)
        elif env_path:
            resolved = Path(env_path)
        else:
            project_root = Path.cwd().resolve()
            project_name = project_root.name or "project"
            date_tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            resolved = project_root / ".mana_llm_logs" / f"{date_tag}-{project_name}-runs.jsonl"
        self.log_file = Path(resolved).resolve()

    def log(self, payload: dict[str, Any]) -> None:
        ensure_dir(self.log_file.parent)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
