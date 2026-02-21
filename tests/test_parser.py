from pathlib import Path

from mana_analyzer.parsers.python_parser import PythonParser


def test_parser_extracts_symbols_and_lines() -> None:
    fixture = Path(__file__).parent / "fixtures" / "sample_project" / "good_module.py"
    parser = PythonParser()

    symbols = parser.parse_file(fixture)
    names = [symbol.name for symbol in symbols]

    assert "good_module.py" in names
    assert "add" in names
    assert "Greeter" in names
    add_symbol = next(item for item in symbols if item.name == "add")
    assert add_symbol.start_line > 0
    assert add_symbol.end_line >= add_symbol.start_line
    assert "def add" in add_symbol.signature
