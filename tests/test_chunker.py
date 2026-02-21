from mana_analyzer.analysis.chunker import CodeChunker
from mana_analyzer.analysis.models import CodeSymbol


def test_chunker_splits_with_overlap() -> None:
    symbol = CodeSymbol(
        kind="function",
        name="big",
        signature="def big():",
        docstring="doc",
        file_path="/tmp/a.py",
        start_line=1,
        end_line=200,
        source="x" * 5000,
    )
    chunker = CodeChunker(max_chars=1000, overlap=100)
    chunks = chunker.build_chunks([symbol])

    assert len(chunks) > 1
    assert chunks[0].file_path == "/tmp/a.py"
    assert chunks[0].id != chunks[1].id
