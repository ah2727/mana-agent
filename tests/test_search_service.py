from pathlib import Path

from mana_analyzer.analysis.models import SearchHit
from mana_analyzer.services.search_service import SearchService


class FakeStore:
    def search(self, _index_dir: Path, query: str, k: int) -> list[SearchHit]:
        assert query == "find add"
        assert k == 3
        suffix = Path(_index_dir).name
        score = 0.9 if suffix == "idx-a" else 0.8
        return [
            SearchHit(
                score=score,
                file_path=f"/tmp/{suffix}.py",
                start_line=1,
                end_line=5,
                symbol_name="add",
                snippet="def add(a, b): return a + b",
            )
        ]


def test_search_service_returns_hits() -> None:
    service = SearchService(store=FakeStore())
    hits = service.search(index_dir="/tmp/index", query="find add", k=3)
    assert len(hits) == 1
    assert hits[0].symbol_name == "add"


def test_search_service_search_multi_merges_and_ranks() -> None:
    service = SearchService(store=FakeStore())
    hits, warnings = service.search_multi(
        index_dirs=[Path("/tmp/idx-b"), Path("/tmp/idx-a")],
        query="find add",
        k=3,
    )
    assert not warnings
    assert len(hits) == 2
    assert hits[0].file_path == "/tmp/idx-a.py"
