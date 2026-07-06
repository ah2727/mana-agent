from __future__ import annotations

import io
import json
import zipfile

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from mana_agent.api.app import create_app
from mana_agent.commands.cli import app as cli_app


def _zip_bytes(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_analyze_zip_returns_downloadable_result_zip() -> None:
    client = TestClient(create_app())
    payload = _zip_bytes(
        {
            "demo/pyproject.toml": "[project]\nname = 'demo'\ndependencies = ['requests']\n",
            "demo/pkg/__init__.py": "",
            "demo/pkg/app.py": "import requests\n\n\ndef run():\n    return 'ok'\n",
        }
    )

    response = client.post(
        "/api/v1/analyze",
        files={"file": ("project.zip", payload, "application/zip")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert "mana-agent-analysis-result.zip" in response.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = set(archive.namelist())
        assert {"analysis-report.md", "analysis-report.json", "manifest.json"} <= names
        report = json.loads(archive.read("analysis-report.json"))
        manifest = json.loads(archive.read("manifest.json"))
    assert report["status"] == "success"
    assert report["project_name"] == "demo"
    assert manifest["input_filename"] == "project.zip"
    assert manifest["analyze_mode"] == "api_zip"
    assert manifest["extracted_root"] == "demo"


def test_analyze_zip_rejects_non_zip_filename() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/v1/analyze",
        files={"file": ("project.txt", b"not a zip", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Only .zip files are supported."}


def test_analyze_zip_rejects_invalid_zip_content() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/v1/analyze",
        files={"file": ("project.zip", b"not a zip", "application/zip")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Uploaded file is not a valid ZIP archive."}


def test_analyze_zip_rejects_parent_traversal() -> None:
    client = TestClient(create_app())
    payload = _zip_bytes({"../evil.py": "print('bad')\n"})

    response = client.post(
        "/api/v1/analyze",
        files={"file": ("project.zip", payload, "application/zip")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "ZIP archive contains unsafe paths."}


def test_analyze_zip_rejects_absolute_path() -> None:
    client = TestClient(create_app())
    payload = _zip_bytes({"/tmp/evil.py": "print('bad')\n"})

    response = client.post(
        "/api/v1/analyze",
        files={"file": ("project.zip", payload, "application/zip")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "ZIP archive contains unsafe paths."}


def test_api_command_imports_and_exposes_server_options() -> None:
    result = CliRunner().invoke(cli_app, ["api", "--help"])

    assert result.exit_code == 0
    assert "--host" in result.stdout
    assert "--port" in result.stdout
