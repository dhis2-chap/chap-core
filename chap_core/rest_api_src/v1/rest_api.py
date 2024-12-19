import json
import logging
from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from chap_core.api_types import PredictionRequest, EvaluationResponse
from chap_core.internal_state import Control, InternalState
from chap_core.log_config import initialize_logging
from chap_core.model_spec import ModelSpec
from chap_core.predictor.feature_spec import Feature, all_features
from chap_core.rest_api_src.data_models import FullPredictionResponse
import chap_core.rest_api_src.worker_functions as wf
from chap_core.predictor.model_registry import registry
from chap_core.worker.interface import SeededJob
from chap_core.rest_api_src.celery_tasks import CeleryPool
from chap_core.rest_api_src.v1.routers import crud, analytics
from . import debug, jobs
from ...database.database import create_db_and_tables

initialize_logging(True, "logs/rest_api.log")
logger = logging.getLogger(__name__)
logger.info("Logging initialized")


def get_app():
    app = FastAPI(root_path="/v1")
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

    return app


app = get_app()
app.include_router(crud.router)
app.include_router(analytics.router)
app.include_router(debug.router)
app.include_router(jobs.router)


class State(BaseModel):
    ready: bool
    status: str
    progress: float = 0


internal_state = InternalState(Control({}), {})

state = State(ready=True, status="idle")


class NaiveJob:
    def __init__(self, func, *args, **kwargs):
        # todo: init a root logger to capture all logs from the job
        self._exception_info = ""
        self._result = ""
        self._status = ""
        self._finished = False
        logger.info("Starting naive job")
        try:
            self._result = func(*args, **kwargs)
            self._status = "finished"
            logger.info("Naive job finished successfully")
            self._finished = True
        except Exception as e:
            self._exception_info = str(e)
            logger.info("Naive job failed with exception: %s", e)
            self._status = "failed"
            self._result = ""

    @property
    def id(self):
        return "naive_job"

    @property
    def status(self):
        return self._status

    @property
    def exception_info(self):
        return self._exception_info

    @property
    def progress(self):
        return 1

    @property
    def result(self):
        return self._result

    def cancel(self):
        pass

    @property
    def is_finished(self):
        return self._finished

    def get_logs(self, n_lines: Optional[int]):
        """Retrives logs from the current job"""
        return ""


class NaiveWorker:
    job_class = NaiveJob

    def queue(self, func, *args, **kwargs):
        # return self.job_class(func(*args, **kwargs))
        return self.job_class(func, *args, **kwargs)


# worker = NaiveWorker()
# worker = BGTaskWorker(BackgroundTasks(), internal_state, state)
# worker = RedisQueue()
worker = CeleryPool()


def set_cur_response(response):
    state["response"] = response


@app.get("favicon.ico")
async def favicon() -> FileResponse:
    return FileResponse("chap_icon.jpeg")


@app.post("/predict")
async def predict(data: PredictionRequest) -> dict:
    """
    Start a prediction task using the given data as training data.
    Results can be retrieved using the get-results endpoint.
    """
    # logger.info(f"Predicting. Worker is {worker}. Data: {data['model']}")
    # dataset = wf.dataset_from_request_v1(data)
    try:
        health_data = wf.get_health_dataset(data)
        target_id = wf.get_target_id(data, ["disease", "diseases", "disease_cases"])
        job = worker.queue(wf.predict_pipeline_from_health_data, health_data.model_dump(), data.estimator_id,
                           data.n_periods, target_id)
        internal_state.current_job = job
    except Exception as e:
        logger.error("Failed to run predic. Exception: %s", e)
        return "{status: 'failed', exception: %s}" % e

    return {"status": "success"}


@app.post("/evaluate")
async def evaluate(data: PredictionRequest, n_splits: Optional[int] = 2, stride: int = 1) -> dict:
    """
    Start an evaluation task using the given data as training data.
    Results can be retrieved using the get-results endpoint.
    """
    json_data = data.model_dump()
    str_data = json.dumps(json_data)
    job = worker.queue(wf.evaluate, str_data, n_splits, stride)
    internal_state.current_job = job
    return {"status": "success",
            "task_id": job.id}


@app.get("/list-models")
async def list_models() -> list[ModelSpec]:
    """
    List all available models. These are not validated. Should set up test suite to validate them
    """
    return registry.list_specifications()


@app.get("/jobs/{job_id}/logs")
async def get_logs(job_id: str, n_lines: Optional[int] = None) -> str:
    """
    Retrieve logs from a job
    """
    job = worker.get_job(job_id)
    return job.get_logs(n_lines)


@app.get("/list-features")
async def list_features() -> list[Feature]:
    """
    List all available features
    """
    return all_features


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
    return cur_job.exception_info or ''


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
        logs=""  # get_logs() # todo: fix
    )


@app.on_event("startup")
def on_startup():
    logger.info("Starting up")
    create_db_and_tables()


def seed(data):
    internal_state.current_job = SeededJob(result=data)


def get_openapi_schema():
    return app.openapi()


def main_backend(seed_data=None):
    import uvicorn
    if seed_data is not None:
        seed(seed_data)
    uvicorn.run(app, host="0.0.0.0", port=8000)
