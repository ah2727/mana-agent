from __future__ import annotations

from mana_analyzer.services.parsers.dart_parser import parse_dart_module
from mana_analyzer.services.parsers.js_ts_parser import parse_js_ts_module
from mana_analyzer.services.parsers.python_parser import parse_python_module


def test_parse_python_module_extracts_symbols(tmp_path) -> None:
    project_root = tmp_path
    module = tmp_path / "a.py"
    module.write_text(
        """
import os

class Foo:
    x = 1


def run():
    return 1
""".strip(),
        encoding="utf-8",
    )

    parsed = parse_python_module(module, project_root)
    assert "os" in parsed.imports
    assert "run" in parsed.functions
    assert any(item.name == "Foo" for item in parsed.classes)


def test_parse_js_ts_module_extracts_imports_and_exports(tmp_path) -> None:
    project_root = tmp_path
    module = tmp_path / "a.ts"
    module.write_text(
        """
import React from 'react'
export function run() {}
export class Service {}
""".strip(),
        encoding="utf-8",
    )

    parsed = parse_js_ts_module(module, project_root)
    assert "react" in parsed.imports
    assert "run" in parsed.functions
    assert any(item.name == "Service" for item in parsed.classes)


def test_parse_dart_module_extracts_symbols(tmp_path) -> None:
    project_root = tmp_path
    module = tmp_path / "a.dart"
    module.write_text(
        """
import 'package:flutter/widgets.dart';
class WidgetBox {}
String buildLabel() {
  return 'ok';
}
""".strip(),
        encoding="utf-8",
    )

    parsed = parse_dart_module(module, project_root)
    assert any(item.startswith("package") for item in parsed.imports)
    assert "buildLabel" in parsed.functions
    assert any(item.name == "WidgetBox" for item in parsed.classes)
