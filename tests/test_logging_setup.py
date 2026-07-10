from __future__ import annotations

import logging
from pathlib import Path
import sys

from mana_agent.utils.logging import setup_logging
from mana_agent.workspaces.paths import mana_home


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
    assert log_file.parent == (mana_home() / "logs")
    assert log_file.name.endswith("-demo-project.log")

    handlers = logging.getLogger().handlers
    assert handlers
    assert all(type(item) is logging.FileHandler for item in handlers)


def test_no_duplicate_log_handlers(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    setup_logging(verbose=True, log_dir=tmp_path / "logs")
    setup_logging(verbose=True, log_dir=tmp_path / "logs")

    handlers = logging.getLogger().handlers
    file_handlers = [item for item in handlers if isinstance(item, logging.FileHandler)]
    stream_handlers = [item for item in handlers if type(item) is logging.StreamHandler]
    assert len(file_handlers) == 1
    assert len(stream_handlers) == 1
    assert logging.getLogger().propagate is False
