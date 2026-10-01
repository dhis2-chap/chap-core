import logging
import os
import traceback

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from chap_core import __version__ as chap_core_version
from chap_core.rest_api.auth import require_api_token, require_api_token_or_registration_key, warn_on_weak_token
from chap_core.rest_api.common_routes import open_router as open_common_router
from chap_core.rest_api.common_routes import router as common_router
from chap_core.rest_api.v1.rest_api import router as v1_router
from chap_core.rest_api.v2.rest_api import router as v2_router

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


app = FastAPI(
    title="CHAP Core API",
    description=api_description,
    version=chap_core_version,
    openapi_tags=openapi_tags,
    root_path=os.environ.get("CHAP_ROOT_PATH", ""),
)

origins = [
    "*",
    "http://localhost:3000",
    "localhost:3000",
]

warn_on_weak_token()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions and log full traceback"""
    logger.error(f"Unhandled exception on {request.method} {request.url.path}")
    logger.error(f"Exception type: {type(exc).__name__}")
    logger.error(f"Exception message: {exc!s}")
    logger.error("Full traceback:")
    logger.error(traceback.format_exc())

    return JSONResponse(
        status_code=500, content={"detail": "Internal server error", "error": str(exc), "type": type(exc).__name__}
    )


# CHAP_API_TOKEN is enforced per router (see rest_api/auth.py). Every router but the open
# health and info endpoints takes it; the v2 router holds only the service registry and its
# proxy, which also take the registration key.
app.include_router(open_common_router)
app.include_router(common_router, dependencies=[Depends(require_api_token)])
app.include_router(v1_router, prefix="/v1", dependencies=[Depends(require_api_token)])
app.include_router(v2_router, prefix="/v2", dependencies=[Depends(require_api_token_or_registration_key)])


def get_openapi_schema():
    return app.openapi()
