from pathlib import Path

from mana_analyzer.parsers.multi_parser import MultiLanguageParser


def test_multi_parser_extracts_js_ts_symbols(tmp_path: Path) -> None:
    module = tmp_path / "feature.ts"
    module.write_text(
        """
export class AuthService {}
export async function login(user: string) { return user }
export const VERSION = "1.0"
""".strip(),
        encoding="utf-8",
    )

    parser = MultiLanguageParser()
    symbols = parser.parse_file(module)
    names = {item.name for item in symbols}

    assert "feature.ts" in names
    assert "AuthService" in names
    assert "login" in names
    assert "VERSION" in names


def test_multi_parser_extracts_dart_symbols(tmp_path: Path) -> None:
    module = tmp_path / "feature.dart"
    module.write_text(
        """
class WidgetBox {}
String buildLabel() {
  return 'ok';
}
final token = 'a';
""".strip(),
        encoding="utf-8",
    )

    parser = MultiLanguageParser()
    symbols = parser.parse_file(module)
    names = {item.name for item in symbols}

    assert "feature.dart" in names
    assert "WidgetBox" in names
    assert "buildLabel" in names
    assert "token" in names
