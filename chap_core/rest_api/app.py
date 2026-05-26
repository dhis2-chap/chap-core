import logging
import os
import traceback

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from chap_core import __version__ as chap_core_version
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
    {"name": "Visualizations", "description": "Generate plots and charts"},
    {"name": "Jobs", "description": "Monitor and manage async jobs"},
    {"name": "Services", "description": "Service registry (v2)"},
]


api_description = """\
**Chap** is a Climate & Health Modeling Platform that brings together
disease forecasting models into a unified ecosystem, connecting researchers
with cutting-edge epidemiological models to policy makers and health
practitioners.

## What it does

- Makes sophisticated modeling workflows more accessible.
- Performs automated, rigorous model evaluation.
- Supplies broad generic functionality for modelers.
- Provides direct integration with [DHIS2](https://dhis2.org).

## Concepts

- **Datasets** — observation series (disease cases, climate covariates,
  population) attached to org-unit polygons. Imported once, reused.
- **Model templates** are the registered model implementations. A
  **configured model** binds a template to a specific set of options
  (lags, precision, covariates, ...).
- **Backtests** evaluate a configured model on historical data and produce
  scored forecasts you can plot, export, and compare.
- **Predictions** are forecasts of future periods produced by a configured
  model, queryable as quantile series.
- **Prediction setups** promote a backtest into a reusable forecast config —
  run it ad-hoc against fresh data, or on a cron schedule.

## Asynchronous work

`POST` endpoints that train, score, or import (most of `/v1/analytics/*`,
`POST /v1/crud/backtests`, `POST /v1/crud/datasets`,
`POST /v1/crud/prediction-setups/{id}/run`) return a `JobResponse` with an
id. Poll `GET /v1/jobs/{id}` for status; once it finishes, fetch the
produced row directly via `GET /v1/jobs/{id}/prediction_result`,
`/evaluation_result`, or `/database_result`.

## Further reading

- [Platform documentation](https://chap.dhis2.org/chap-modeling-platform/)
- Contact: <chap@dhis2.org>
"""


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


app.include_router(common_router)
app.include_router(v1_router, prefix="/v1")
app.include_router(v2_router, prefix="/v2")


def get_openapi_schema():
    return app.openapi()
