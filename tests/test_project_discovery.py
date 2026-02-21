from __future__ import annotations

import json
from pathlib import Path

from mana_analyzer.utils.project_discovery import discover_subprojects


def test_discover_subprojects_finds_nested_manifests(tmp_path: Path) -> None:
    (tmp_path / "app").mkdir(parents=True)
    (tmp_path / "backend").mkdir(parents=True)
    (tmp_path / "website").mkdir(parents=True)

    (tmp_path / "app" / "pubspec.yaml").write_text("name: app\n", encoding="utf-8")
    (tmp_path / "backend" / "nest-cli.json").write_text("{}", encoding="utf-8")
    (tmp_path / "backend" / "package.json").write_text(json.dumps({"dependencies": {"@nestjs/core": "^10"}}), encoding="utf-8")
    (tmp_path / "website" / "package.json").write_text(json.dumps({"dependencies": {"react": "^18", "vite": "^5"}}), encoding="utf-8")
    (tmp_path / "website" / "vite.config.ts").write_text("export default {};\n", encoding="utf-8")

    projects = discover_subprojects(tmp_path)

    roots = {str(item.root_path.relative_to(tmp_path)) for item in projects}
    assert "app" in roots
    assert "backend" in roots
    assert "website" in roots

    backend = next(item for item in projects if item.root_path == (tmp_path / "backend"))
    assert "NestJS" in backend.framework_hints
    assert "npm" in backend.package_managers

    website = next(item for item in projects if item.root_path == (tmp_path / "website"))
    assert "Vite" in website.framework_hints
