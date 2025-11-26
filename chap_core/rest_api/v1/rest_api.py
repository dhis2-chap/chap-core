import logging
import traceback

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, ORJSONResponse
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel

from chap_core.api_types import EvaluationResponse
from chap_core.internal_state import Control, InternalState
from chap_core.log_config import initialize_logging
from chap_core.model_spec import ModelSpec
from chap_core.predictor.feature_spec import Feature
from chap_core.rest_api.celery_tasks import CeleryPool
from chap_core.rest_api.data_models import FullPredictionResponse
from chap_core.rest_api.v1.routers import analytics, crud, visualization
from chap_core.worker.interface import SeededJob

from . import debug, jobs
from .routers.dependencies import get_settings

initialize_logging(True, "logs/rest_api.log")
logger = logging.getLogger(__name__)
logger.info("Logging initialized")


# Job id and database id


def create_api():
    app = FastAPI(
        root_path="/v1",
        default_response_class=ORJSONResponse,
    )

    origins = [
        "*",  # Allow all origins
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

    # Global exception handler to log full stack traces
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Catch all unhandled exceptions and log full traceback"""
        # Handle InvalidVersion exceptions without full stack trace
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

    return app


app = create_api()
app.include_router(crud.router)
app.include_router(analytics.router)
app.include_router(debug.router)
app.include_router(jobs.router)
app.include_router(visualization.router)


class State(BaseModel):
    ready: bool
    status: str
    progress: float = 0


internal_state = InternalState(Control({}), {})

state = State(ready=True, status="idle")


worker = CeleryPool()


def set_cur_response(response):
    state["response"] = response


@app.get("favicon.ico")
async def favicon() -> FileResponse:
    return FileResponse("chap_icon.jpeg")


@app.get("/list-models", deprecated=True)
async def list_models() -> list[ModelSpec]:
    """
    List all available models. These are not validated. Should set up test suite to validate them
    """
    return []


# @app.get("/jobs/{job_id}/logs")
# async def get_logs(job_id: str, n_lines: Optional[int] = None) -> str:
#     """
#     Retrieve logs from a job
#     """
#     job = worker.get_job(job_id)
#     return job.get_logs(n_lines)


@app.get("/list-features", deprecated=True)
async def list_features() -> list[Feature]:
    """
    List all available features
    """
    return []
    # return all_features


@app.get("/get-results")
async def get_results() -> FullPredictionResponse:
    """
    Retrieve results made by the model
    """
    cur_job = internal_state.current_job
    if cur_job.status == "failed":
        raise HTTPException(status_code=400, detail="Job failed. Check the exception endpoint for more information")

    if not (cur_job and cur_job.is_finished):
        raise HTTPException(status_code=400, detail="No response available")
    result = cur_job.result
    return result


@app.get("/get-evaluation-results")
async def get_evaluation_results() -> EvaluationResponse:
    """
    Retrieve evaluation results made by the model
    """
    cur_job = internal_state.current_job
    if cur_job.status == "failed":
        raise HTTPException(status_code=400, detail="Job failed. Check the exception endpoint for more information")

    if not (cur_job and cur_job.is_finished):
        raise HTTPException(status_code=400, detail="No response available")
    return cur_job.result


@app.get("/get-exception")
async def get_exception() -> str:
    """
    Retrieve exception information if the job failed
    """
    cur_job = internal_state.current_job
    return cur_job.exception_info or ""


@app.post("/cancel")
async def cancel() -> dict:
    """
    Cancel the current training
    """
    if internal_state.control is not None:
        internal_state.control.cancel()
    return {"status": "success"}


@app.get("/status")
async def get_status() -> State:
    """
    Retrieve the current status of the model
    """
    if internal_state.is_ready():
        return State(ready=True, status="idle")

    return State(
        ready=False,
        status=internal_state.current_job.status,
        progress=internal_state.current_job.progress,
        logs="",  # get_logs() # todo: fix
    )


class HealthResponse(BaseModel):
    status: str
    message: str


@app.get("/health")
async def health(worker_config=Depends(get_settings)) -> HealthResponse:
    # try:
    #     wf.initialize_gee_client(usecwd=True, worker_config=worker_config)
    # except GEEError as e:
    #     return HealthResponse(status="failed", message="GEE authentication might not be set up properly: " + str(e))
    return HealthResponse(status="success", message="healthy")


@app.get("/version")
async def version() -> dict:
    """
    Retrieve the current version of the API
    """
    # read version from init
    from chap_core import __version__ as chap_core_version

    return {"version": chap_core_version}


class CompatibilityResponse(BaseModel):
    compatible: bool
    description: str


class SystemInfoResponse(BaseModel):
    chap_core_version: str
    python_version: str
    os: str


@app.get("/is-compatible")
async def is_compatible(modelling_app_version: str) -> CompatibilityResponse:
    """
    Check if the modelling app version is compatible with the current API version
    """

    # new: Hardcoded minimum version to allow more easy update of frontend
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
    else:
        return CompatibilityResponse(
            compatible=True,
            description=f"Modelling app version {modelling_app_version} is compatible with the current API version {chap_core_version}",
        )

    """
    # read version from init (add random string to avoid github caching)
    random_string = str(random.randint(0, 10000000000))
    compatibility_file = f"https://raw.githubusercontent.com/dhis2-chap/versioning/refs/heads/main/modelling-app-chap-core.yml?r={random_string}"
    response = requests.get(compatibility_file)
    if response.status_code != 200:
        return CompatibilityResponse(compatible = False, description="Could not load compatibility file")

    # parse yaml
    compatibility_data = yaml.safe_load(response.text)
    modelling_app_versions = list(compatibility_data.keys())
    if modelling_app_version not in modelling_app_versions:
        return CompatibilityResponse(compatible = False, description = f"Modelling app version {modelling_app_version} not found in compatibility file, which contains {modelling_app_versions}")

    if chap_core_version not in compatibility_data[modelling_app_version]:
        description = f"Modelling app version {modelling_app_version} is not compatible with chap core version {chap_core_version}. Supported versions are {compatibility_data[modelling_app_version]}."
        is_compatible = False
    else:
        description = f"Modelling app version {modelling_app_version} is compatible with chap core version {chap_core_version}. The supported versions are {', '.join(compatibility_data[modelling_app_version])}."
        is_compatible = True

    return CompatibilityResponse(
        compatible=is_compatible, description=description)
    """


@app.get("/system-info")
async def system_info() -> SystemInfoResponse:
    """
    Retrieve system information
    """
    import platform

    from chap_core import __version__ as chap_core_version

    return SystemInfoResponse(
        chap_core_version=chap_core_version, python_version=platform.python_version(), os=platform.platform()
    )


def seed(data):
    internal_state.current_job = SeededJob(result=data)


def get_openapi_schema():
    return app.openapi()


def main_backend(seed_data=None, auto_reload=False):
    import uvicorn
    from chap_core.database.database import create_db_and_tables

    create_db_and_tables()

    if seed_data is not None:
        seed(seed_data)

    if auto_reload:
        app_path = "chap_core.rest_api.v1.rest_api:app"
        uvicorn.run(app_path, host="0.0.0.0", port=8000, reload=auto_reload)
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)
