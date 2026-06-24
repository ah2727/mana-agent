from __future__ import annotations

from pathlib import Path

from mana_agent.commands import cli_internal


class _FakeIndexService:
    def __init__(self, *, fail_vectors: bool = False, fail_all: bool = False) -> None:
        self.fail_vectors = fail_vectors
        self.fail_all = fail_all
        self.calls: list[bool] = []

    def index(self, *, target_path, index_dir, rebuild=False, vectors=True):
        self.calls.append(vectors)
        if self.fail_all:
            raise RuntimeError("indexing unavailable")
        if vectors and self.fail_vectors:
            raise RuntimeError("no embeddings backend")
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        (Path(index_dir) / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
        return {}


def _patch_builder(monkeypatch, service: _FakeIndexService) -> None:
    monkeypatch.setattr(
        "mana_agent.commands.cli.build_index_service",
        lambda _s: service,
        raising=False,
    )


def test_background_index_builds_and_marks_ready(monkeypatch, tmp_path: Path) -> None:
    service = _FakeIndexService()
    _patch_builder(monkeypatch, service)
    index_dir = tmp_path / ".mana" / "index"
    state: dict[str, object] = {"status": "idle", "announced": False, "error": ""}

    thread = cli_internal._start_background_index(
        settings=object(),
        target_root=tmp_path,
        index_dir=index_dir,
        state=state,
    )
    thread.join(timeout=10)

    assert state["status"] == "ready"
    assert (index_dir / "chunks.jsonl").exists()


def test_background_index_falls_back_to_chunks_only(monkeypatch, tmp_path: Path) -> None:
    service = _FakeIndexService(fail_vectors=True)
    _patch_builder(monkeypatch, service)
    index_dir = tmp_path / ".mana" / "index"
    state: dict[str, object] = {"status": "idle", "announced": False, "error": ""}

    thread = cli_internal._start_background_index(
        settings=object(),
        target_root=tmp_path,
        index_dir=index_dir,
        state=state,
    )
    thread.join(timeout=10)

    assert state["status"] == "ready"
    # vectors=True attempted first, then chunks-only (vectors=False).
    assert service.calls == [True, False]


def test_background_index_marks_failed(monkeypatch, tmp_path: Path) -> None:
    service = _FakeIndexService(fail_all=True)
    _patch_builder(monkeypatch, service)
    index_dir = tmp_path / ".mana" / "index"
    state: dict[str, object] = {"status": "idle", "announced": False, "error": ""}

    thread = cli_internal._start_background_index(
        settings=object(),
        target_root=tmp_path,
        index_dir=index_dir,
        state=state,
    )
    thread.join(timeout=10)

    assert state["status"] == "failed"
    assert state["error"]
