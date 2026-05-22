"""Common API endpoints shared across all API versions.

These endpoints are mounted at the root level of the application and provide
version-independent functionality such as health checks and system information.
"""

import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import text

from chap_core.database.database import engine
from chap_core.rest_api.celery_tasks import app as celery_app
from chap_core.rest_api.v1.routers.dependencies import get_settings
from chap_core.util import load_redis

logger = logging.getLogger(__name__)


# --- Response models ---


class HealthResponse(BaseModel):
    """Liveness check response indicating the API process is reachable."""

    status: str = Field(description="Always `success` when the API is reachable.", examples=["success"])
    message: str = Field(description="Human-readable status message.", examples=["healthy"])


class ReadinessResponse(BaseModel):
    """Readiness check response with per-dependency status."""

    status: str = Field(
        description="Overall readiness state. `success` if every dependency is reachable, otherwise `unhealthy`.",
        examples=["success"],
    )
    checks: dict[str, str] = Field(
        description=(
            "Per-dependency status keyed by dependency name (`db`, `redis`, `celery`). "
            "Each value is `ok` on success, or `error: <reason>` on failure."
        ),
        examples=[{"db": "ok", "redis": "ok", "celery": "ok"}],
    )


class SystemInfoResponse(BaseModel):
    """System information response."""

    chap_core_version: str = Field(description="Installed `chap_core` package version.")
    python_version: str = Field(description="Python runtime version of the API process.")
    server_date: str = Field(description="Current server time as an ISO-8601 UTC string.")
    server_time_zone_id: str = Field(description="Server timezone identifier (always `Etc/UTC`).")
    revision: str = Field(description="Git revision the running image was built from (empty if unknown).")


# --- Router ---

router = APIRouter(tags=["System"])


# -- Health and info endpoints --


@router.get(
    "/health",
    summary="Liveness probe",
    description=(
        "Cheap liveness check that returns immediately if the API process is reachable. "
        "Does not verify downstream dependencies — use `/health/ready` for that."
    ),
    response_description="API process is alive.",
)
async def health(worker_config=Depends(get_settings)) -> HealthResponse:
    return HealthResponse(status="success", message="healthy")


def _check_db() -> str:
    if engine is None:
        return "error: database not configured"
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return "ok"
    except Exception as exc:
        logger.warning("Readiness DB check failed", exc_info=True)
        return f"error: {exc.__class__.__name__}"


def _check_redis() -> str:
    try:
        load_redis().ping()
        return "ok"
    except Exception as exc:
        logger.warning("Readiness Redis check failed", exc_info=True)
        return f"error: {exc.__class__.__name__}"


def _check_celery() -> str:
    try:
        replies = celery_app.control.ping(timeout=1.0)
        if not replies:
            return "error: no workers responded"
        return "ok"
    except Exception as exc:
        logger.warning("Readiness Celery check failed", exc_info=True)
        return f"error: {exc.__class__.__name__}"


def _check_celery_db_roundtrip() -> str:
    """End-to-end probe: dispatch `chap.db_roundtrip`, wait for the reply.

    Verifies the full API → broker → worker → DB → result-backend → API path.
    Slower than `_check_celery` (which only confirms worker liveness); intended
    for the on-demand `/health/probe` endpoint, not for hot-path readiness.
    """
    result = celery_app.send_task("chap.db_roundtrip")
    try:
        try:
            reply = result.get(timeout=2.0, propagate=False)
        except Exception as exc:
            logger.warning("Health probe round-trip raised", exc_info=True)
            return f"error: {exc.__class__.__name__}"
        if isinstance(reply, BaseException):
            return f"error: {reply.__class__.__name__}"
        if reply == "ok":
            return "ok"
        return f"error: unexpected reply {reply!r}"
    finally:
        result.forget()


@router.get(
    "/health/ready",
    summary="Readiness probe with dependency checks",
    description=(
        "Synchronously verifies that the required infrastructure dependencies are reachable:\n\n"
        "- **db**: executes `SELECT 1` against PostgreSQL.\n"
        "- **redis**: issues a Redis `PING`.\n"
        "- **celery**: broadcasts a worker `ping` (1s timeout) and requires at least one reply.\n\n"
        "Returns `200` with `status: success` when all checks pass, or `503` with `status: unhealthy` "
        "and a per-dependency `checks` map so dashboards can identify which dependency is down. "
        "Intended for use as a Kubernetes readiness probe or load balancer health check."
    ),
    response_model=ReadinessResponse,
    response_description="All dependencies are reachable.",
    responses={
        503: {
            "description": "One or more dependencies are unreachable.",
            "model": ReadinessResponse,
        },
    },
)
async def readiness() -> Response:
    checks = {
        "db": _check_db(),
        "redis": _check_redis(),
        "celery": _check_celery(),
    }
    healthy = all(v == "ok" for v in checks.values())
    body = ReadinessResponse(status="success" if healthy else "unhealthy", checks=checks)
    status_code = 200 if healthy else 503
    return JSONResponse(status_code=status_code, content=body.model_dump())


@router.get(
    "/health/probe",
    summary="End-to-end celery + DB round-trip probe",
    description=(
        "Synchronously dispatches a task to a celery worker, requires the worker to run "
        "`SELECT 1` against PostgreSQL, and waits up to 2 seconds for the result to come back "
        "via the result backend. Verifies the full pipeline that `/health/ready` only checks "
        "shallowly:\n\n"
        "- API → broker enqueue\n"
        "- Worker pickup and execution\n"
        "- Worker → DB read\n"
        "- Result backend write/read\n\n"
        "Slower than `/health/ready` (tens of ms when healthy, up to 2s on timeout). "
        "**Not** intended for Kubernetes readiness probes — use this from dashboards or "
        "on-demand ops checks."
    ),
    response_model=ReadinessResponse,
    response_description="End-to-end round-trip succeeded.",
    responses={
        503: {
            "description": "Celery round-trip failed or timed out.",
            "model": ReadinessResponse,
        },
    },
)
async def deep_probe() -> Response:
    roundtrip = _check_celery_db_roundtrip()
    healthy = roundtrip == "ok"
    body = ReadinessResponse(
        status="success" if healthy else "unhealthy",
        checks={"celery_db_roundtrip": roundtrip},
    )
    return JSONResponse(status_code=200 if healthy else 503, content=body.model_dump())


@router.get(
    "/system/info",
    summary="System and version information",
    description="Returns build/runtime metadata (CHAP version, Python version, server time, git revision).",
)
async def system_info() -> SystemInfoResponse:
    import platform

    from chap_core import __version__ as chap_core_version

    server_date = datetime.now(UTC).isoformat()
    revision = os.environ.get("GIT_REVISION", "")

    return SystemInfoResponse(
        chap_core_version=chap_core_version,
        python_version=platform.python_version(),
        server_date=server_date,
        server_time_zone_id="Etc/UTC",
        revision=revision,
    )


# -- Static assets --


@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    path = Path("chap_icon.jpeg")
    if not path.is_file():
        return Response(status_code=204)
    return FileResponse(path)
