from mana_analyzer.commands.cli import _extract_structured_answer


def test_extract_structured_answer_handles_python_list_reasoning_blocks() -> None:
    raw = (
        "[{'id': 'rs_1', 'summary': [], 'type': 'reasoning'}, "
        "{'type': 'text', 'text': 'Clean final answer text'}]"
    )
    answer, payload = _extract_structured_answer(raw)
    assert answer == "Clean final answer text"
    assert payload is None

