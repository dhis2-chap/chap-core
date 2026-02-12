import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, ORJSONResponse
from packaging.version import InvalidVersion

from chap_core.common_routes import router as common_router
from chap_core.rest_api.v1.rest_api import router as v1_router
from chap_core.rest_api.v2.rest_api import router as v2_router

logger = logging.getLogger(__name__)

app = FastAPI(
    title="CHAP Core API",
    default_response_class=ORJSONResponse,
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
    if isinstance(exc, InvalidVersion):
        logger.warning(f"Invalid version string on {request.method} {request.url.path}: {str(exc)}")
    else:
        logger.error(f"Unhandled exception on {request.method} {request.url.path}")
        logger.error(f"Exception type: {type(exc).__name__}")
        logger.error(f"Exception message: {str(exc)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())

    return JSONResponse(
        status_code=500, content={"detail": "Internal server error", "error": str(exc), "type": type(exc).__name__}
    )


app.include_router(common_router)
app.include_router(common_router, prefix="/v1", include_in_schema=False)
app.include_router(v1_router, prefix="/v1")
app.include_router(v2_router, prefix="/v2")
