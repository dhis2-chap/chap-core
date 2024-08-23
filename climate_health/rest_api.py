import json
from contextlib import asynccontextmanager
import logging
from asyncio import CancelledError
from typing import List, Union

from fastapi import BackgroundTasks, UploadFile, HTTPException
from pydantic import BaseModel
# from fastapi.responses import HTMLResponse

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from climate_health.api import read_zip_folder, train_on_prediction_data
from climate_health.api_types import RequestV1
from climate_health.google_earth_engine.gee_era5 import Era5LandGoogleEarthEngine
from climate_health.internal_state import Control, InternalState
from climate_health.model_spec import ModelSpec, model_spec_from_model
from climate_health.predictor import all_models
from climate_health.predictor.feature_spec import Feature, all_features
from climate_health.rest_api_src.worker_functions import train_on_zip_file, train_on_json_data
from climate_health.training_control import TrainingControl
from dotenv import load_dotenv, find_dotenv

from climate_health.worker.background_tasks_worker import BGTaskWorker
from climate_health.worker.rq_worker import RedisQueue

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_app():
    app = FastAPI(
        root_path="/v1"
    )
    origins = [
        '*',  # Allow all origins
        "http://localhost:3000",
        "localhost:3000",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    return app


app = get_app()


class State(BaseModel):
    ready: bool
    status: str
    progress: float = 0


internal_state = InternalState(Control({}), {})

state = State(ready=True, status='idle')


class NaiveWorker:
    def queue(self, func, *args, **kwargs):
        return NaiveJob(func(*args, **kwargs))


class NaiveJob:
    def __init__(self, result):
        self._result = result

    @property
    def status(self):
        return 'finished'

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
        return True


# worker = NaiveWorker()
# worker = BGTaskWorker(BackgroundTasks(), internal_state, state)
worker = RedisQueue()


def set_cur_response(response):
    state['response'] = response


class PredictionResponse(BaseModel):
    value: float
    orgUnit: str
    dataElement: str
    period: str


class FullPredictionResponse(BaseModel):
    diseaseId: str
    dataValues: List[PredictionResponse]


@app.get('favicon.ico')
async def favicon() -> FileResponse:
    return FileResponse('chap_icon.jpeg')


@app.post('/set-model-path')
async def set_model_path(model_path: str) -> dict:
    '''
    Set the model to be used for training and evaluation
    https://github.com/knutdrand/external_rmodel_example.git
    '''
    internal_state.model_path = model_path
    return {'status': 'success'}


@app.post('/zip-file/')
async def post_zip_file(file: Union[UploadFile, None] = None, background_tasks: BackgroundTasks = None) -> dict:
    '''
    Post a zip file containing the data needed for training and evaluation, and start the training
    '''

    # Herman: I comment these two lines out, since we accept more than one jobs for now?
    # if not internal_state.is_ready():
    #    raise HTTPException(status_code=400, detail="Model is currently training")

    model_name, model_path = 'HierarchicalModel', None
    if internal_state.model_path is not None:
        model_name = 'external'
        model_path = internal_state.model_path

    job = worker.queue(train_on_zip_file, file, model_name, model_path)
    internal_state.current_job = job

    return {'status': 'success'}

@app.post('/predict-from-json/')
async def predict_from_json(data: RequestV1) -> dict:
    '''
    Post a json file containing the data needed for prediction
    '''
    model_name, model_path = 'HierarchicalModel', None
    if internal_state.model_path is not None:
        model_name = 'external'
        model_path = internal_state.model_path
    json_data = data.model_dump()

    str_data = json.dumps(json_data)

    job = worker.queue(train_on_json_data, str_data, model_name, model_path)
    internal_state.current_job = job

    return {'status': 'success'}

    #if internal_state.model_path is not None:
    #    model_name = 'external'
    #    model_path = internal_state.model_path
    #response = train_on_prediction_data(data, model_name, model_path)
    #return response


@app.get('/list-models')
async def list_models() -> list[ModelSpec]:
    '''
    List all available models. These are not validated. Should set up test suite to validate them
    '''
    model_list = (model_spec_from_model(model) for model in all_models)
    valid_model_list = [m for m in model_list if m is not None]
    return valid_model_list


@app.get('/list-features')
async def list_features() -> list[Feature]:
    '''
    List all available features
    '''
    return all_features


@app.get('/get-results')
async def get_results() -> FullPredictionResponse:
    '''
    Retrieve results made by the model
    '''
    cur_job = internal_state.current_job
    print(cur_job)
    if not (cur_job and cur_job.is_finished):
        raise HTTPException(status_code=400, detail="No response available")
    result = cur_job.result
    print(result)
    return result


@app.post('/cancel')
async def cancel() -> dict:
    '''
    Cancel the current training
    '''
    if internal_state.control is not None:
        internal_state.control.cancel()
    return {'status': 'success'}


@app.get('/status')
async def get_status() -> State:
    '''
    Retrieve the current status of the model
    '''
    if internal_state.is_ready():
        return State(ready=True, status='idle')

    return State(ready=False,
                 status=internal_state.current_job.status,
                 progress=internal_state.current_job.progress)


def main_backend():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
