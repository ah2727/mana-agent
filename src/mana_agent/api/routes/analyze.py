from __future__ import annotations

from tempfile import TemporaryDirectory

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from mana_agent.api.exceptions import ManaApiError
from mana_agent.api.services.analyze_service import run_zip_analysis
from mana_agent.api.services.zip_service import (
    InvalidZipError,
    UnsafeZipError,
    ZipValidationError,
)

router = APIRouter(prefix="/api/v1", tags=["analyze"])


@router.post("/analyze")
async def analyze_zip(file: UploadFile | None = File(None)) -> FileResponse:
    if file is None or not file.filename:
        raise ManaApiError(422, "Upload field 'file' is required.")

    temp_dir = TemporaryDirectory(prefix="mana_api_analyze_")
    try:
        result_zip = await run_zip_analysis(file=file, workspace_root=temp_dir.name)
    except ZipValidationError as exc:
        temp_dir.cleanup()
        raise ManaApiError(400, str(exc)) from exc
    except InvalidZipError as exc:
        temp_dir.cleanup()
        raise ManaApiError(400, str(exc)) from exc
    except UnsafeZipError as exc:
        temp_dir.cleanup()
        raise ManaApiError(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - keep API failures clean
        temp_dir.cleanup()
        raise ManaApiError(500, "Analyze failed.", error=str(exc)) from exc

    return FileResponse(
        result_zip,
        media_type="application/zip",
        filename="mana-agent-analysis-result.zip",
        background=BackgroundTask(temp_dir.cleanup),
    )
