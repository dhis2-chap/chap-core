"""Common API endpoints shared across all API versions.

These endpoints are mounted at the root level of the application and provide
version-independent functionality such as health checks and system information.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from chap_core.rest_api.v1.routers.dependencies import get_settings

logger = logging.getLogger(__name__)


# --- Response models ---


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    message: str


class SystemInfoResponse(BaseModel):
    """System information response."""

    chap_core_version: str
    python_version: str


# --- Router ---

router = APIRouter(tags=["System"])


# -- Health and info endpoints --


@router.get("/health")
async def health(worker_config=Depends(get_settings)) -> HealthResponse:
    """Check that the API is running and healthy."""
    return HealthResponse(status="success", message="healthy")


@router.get("/system/info")
async def system_info() -> SystemInfoResponse:
    """Return system information including versions."""
    import platform

    from chap_core import __version__ as chap_core_version

    return SystemInfoResponse(chap_core_version=chap_core_version, python_version=platform.python_version())


# -- Static assets --


@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    path = Path("chap_icon.jpeg")
    if not path.is_file():
        return Response(status_code=204)
    return FileResponse(path)
