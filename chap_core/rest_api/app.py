import logging
import uuid
from contextlib import asynccontextmanager

# pandera imports geopandas, and with it pyproj, the first time a schema is built. In a
# sync route that happens on a threadpool thread, and once pyproj has been imported off
# the main thread every later fork() child segfaults on macOS. Import it here instead.
import pyproj  # noqa: F401
from anyio import to_thread
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.authentication import AuthenticationMiddleware

from chap_core import __version__ as chap_core_version
from chap_core.auth.tokens import get_authentication
from chap_core.database.database import DB_POOL_CONNECTIONS
from chap_core.rest_api.auth import ChapTokenBackend, on_authentication_error, warn_on_weak_token
from chap_core.rest_api.authorization import PermissionDenied, check_route_permissions, permission_denied_handler
from chap_core.rest_api.common_routes import router as common_router
from chap_core.rest_api.v1.rest_api import router as v1_router
from chap_core.rest_api.v2.rest_api import router as v2_router
from chap_core.settings import get_settings

logger = logging.getLogger(__name__)

openapi_tags = [
    {"name": "System", "description": "Health checks and system information"},
    {"name": "Backtests", "description": "Create, manage, and query backtests and evaluation results"},
    {"name": "Metrics", "description": "Compute and export evaluation metrics"},
    {"name": "Predictions", "description": "Create, manage, and query predictions"},
    {"name": "Datasets", "description": "Create, manage, and export datasets"},
    {"name": "Models", "description": "Model templates and configured models"},
    {"name": "Prediction Setups", "description": "Automation configs that schedule predictions from a backtest"},
    {"name": "Alerts", "description": "Alerts raised against a policy, and the gate that releases them"},
    {"name": "Alert Policies", "description": "Tiered thresholds a prediction setup raises alerts against"},
    {"name": "Visualizations", "description": "Generate plots and charts"},
    {"name": "Jobs", "description": "Monitor and manage async jobs"},
    {"name": "Services", "description": "Service registry (v2)"},
]


api_description = (
    "Chap is a Climate & Health Modeling Platform that brings together "
    "disease forecasting models into a unified ecosystem, connecting "
    "researchers with cutting-edge epidemiological models to policy "
    "makers and health practitioners."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Refuse to start on a token configuration that cannot work (an unset secret, a short one, a
    # repeat), or with a route that declares no permission. Here rather than at import, so
    # exporting the OpenAPI schema does not need the token secrets.
    warn_on_weak_token()
    get_authentication()
    check_route_permissions(app)
    # Sync handlers run in this threadpool. One thread per pooled database connection, so
    # extra requests wait for a thread instead of failing on a QueuePool timeout.
    to_thread.current_default_thread_limiter().total_tokens = DB_POOL_CONNECTIONS
    yield


app = FastAPI(
    lifespan=lifespan,
    title="CHAP Core API",
    description=api_description,
    version=chap_core_version,
    openapi_tags=openapi_tags,
    root_path=get_settings().root_path,
)

CORS_ORIGINS_ENV_VAR = "CHAP_CORS_ORIGINS"


def get_cors_origins() -> list[str]:
    """Origins allowed to call the API from a browser, from ``CHAP_CORS_ORIGINS``."""
    return get_settings().cors_origins



# Must stay before CORSMiddleware: add_middleware inserts at the front, so this keeps CORS
# outermost and 401 responses still carry CORS headers.
app.add_middleware(AuthenticationMiddleware, backend=ChapTokenBackend(), on_error=on_authentication_error)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    # The API authenticates with a bearer token, not cookies.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Starlette defaults to level 9, which took 2.3 s on a 33 MB facility-scale response.
# Level 4 took 0.3 s for an 8% larger body (CLIM-1876).
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=4)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Log unhandled exceptions with a trace id and return only the trace id, so internals stay out of the response."""
    trace_id = uuid.uuid4().hex
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path} (trace_id={trace_id})",
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error", "trace_id": trace_id})


app.include_router(common_router)
app.include_router(v1_router, prefix="/v1")
app.include_router(v2_router, prefix="/v2")

app.add_exception_handler(PermissionDenied, permission_denied_handler)


def get_openapi_schema():
    return app.openapi()
