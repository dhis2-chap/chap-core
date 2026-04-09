import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from chap_core.rest_api.common_routes import router as common_router
from chap_core.rest_api.v1.rest_api import router as v1_router
from chap_core.rest_api.v2.rest_api import router as v2_router

logger = logging.getLogger(__name__)

openapi_tags = [
    {"name": "System", "description": "Health checks and system information"},
    {"name": "Backtests", "description": "Create, manage, and query backtests and evaluation results"},
    {"name": "Predictions", "description": "Create, manage, and query predictions"},
    {"name": "Datasets", "description": "Create, manage, and export datasets"},
    {"name": "Models", "description": "Model templates and configured models"},
    {"name": "Visualizations", "description": "Generate plots and charts"},
    {"name": "Jobs", "description": "Monitor and manage async jobs"},
    {"name": "Debug", "description": "Debug and diagnostic endpoints"},
    {"name": "Services", "description": "Service registry (v2)"},
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    from chap_core.database.database import create_db_and_tables

    create_db_and_tables()
    yield


app = FastAPI(
    title="CHAP Core API",
    openapi_tags=openapi_tags,
    lifespan=lifespan,
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
