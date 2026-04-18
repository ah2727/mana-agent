from __future__ import annotations

import logging
from pathlib import Path
import sys

from mana_analyzer.utils.logging import setup_logging


def test_setup_logging_writes_file_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    log_file = setup_logging(verbose=True, log_dir=tmp_path / "logs")
    assert log_file.exists()
    assert log_file.parent == (tmp_path / "logs")

    handlers = logging.getLogger().handlers
    assert handlers
    assert any(isinstance(item, logging.FileHandler) for item in handlers)
    stream_handlers = [item for item in handlers if type(item) is logging.StreamHandler]
    assert len(stream_handlers) == 1
    assert stream_handlers[0].stream is sys.stderr


def test_setup_logging_uses_date_project_name(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "demo-project"
    project.mkdir()
    monkeypatch.chdir(project)
    log_file = setup_logging(verbose=False)
    assert log_file.parent == (project / ".mana" / "logs")
    assert log_file.name.endswith("-demo-project.log")

    handlers = logging.getLogger().handlers
    assert handlers
    assert all(type(item) is logging.FileHandler for item in handlers)
