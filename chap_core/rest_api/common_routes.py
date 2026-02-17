"""Common API endpoints shared across all API versions.

These endpoints are mounted at the root level of the application and provide
version-independent functionality such as health checks,
compatibility checks, and system information.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, Response
from packaging.version import Version
from pydantic import BaseModel

from chap_core.rest_api.v1.routers.dependencies import get_settings

logger = logging.getLogger(__name__)


# --- Response models ---


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    message: str


class CompatibilityResponse(BaseModel):
    """Response for modelling app compatibility checks."""

    compatible: bool
    description: str


class SystemInfoResponse(BaseModel):
    """System information response."""

    chap_core_version: str
    python_version: str
    os: str


# --- Router ---

router = APIRouter(tags=["System"])


# -- Health and info endpoints --


@router.get("/health")
async def health(worker_config=Depends(get_settings)) -> HealthResponse:
    """Check that the API is running and healthy."""
    return HealthResponse(status="success", message="healthy")


@router.get("/is-compatible")
async def is_compatible(modelling_app_version: str) -> CompatibilityResponse:
    """Check if a modelling app version is compatible with this API."""
    from chap_core import (
        __minimum_modelling_app_version__ as minimum_modelling_app_version,
    )
    from chap_core import (
        __version__ as chap_core_version,
    )

    if Version(modelling_app_version) < Version(minimum_modelling_app_version):
        return CompatibilityResponse(
            compatible=False,
            description=f"Modelling app version {modelling_app_version} is too old. Minimum version is {minimum_modelling_app_version}",
        )
    return CompatibilityResponse(
        compatible=True,
        description=f"Modelling app version {modelling_app_version} is compatible with the current API version {chap_core_version}",
    )


@router.get("/system-info")
async def system_info() -> SystemInfoResponse:
    """Return system information including versions and OS."""
    import platform

    from chap_core import __version__ as chap_core_version

    return SystemInfoResponse(
        chap_core_version=chap_core_version, python_version=platform.python_version(), os=platform.platform()
    )


# -- Static assets --


@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    path = Path("chap_icon.jpeg")
    if not path.is_file():
        return Response(status_code=204)
    return FileResponse(path)
