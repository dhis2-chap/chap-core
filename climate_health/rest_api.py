from contextlib import asynccontextmanager
import dataclasses
import logging
import time
from asyncio import CancelledError
from enum import Enum
from typing import List, Union, Optional

from fastapi import BackgroundTasks, UploadFile, HTTPException
from pydantic import BaseModel
# from fastapi.responses import HTMLResponse

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from climate_health.api import read_zip_folder, dhis_zip_flow, train_on_prediction_data
from climate_health.dhis2_interface.json_parsing import parse_json_rows
from climate_health.google_earth_engine.gee_era5 import GoogleEarthEngine
from climate_health.model_spec import ModelSpec, model_spec_from_model
from climate_health.predictor import ModelType, get_model, all_model_names, all_models
from climate_health.predictor.feature_spec import Feature, all_features
from climate_health.training_control import TrainingControl
from dotenv import load_dotenv, find_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Control:
    def __init__(self, controls):
        self._controls = controls
        self._status = 'idle'
        self._current_control = None
        self._is_cancelled = False

    @property
    def current_control(self):
        return self._current_control

    def cancel(self):
        if self._current_control is not None:
            self._current_control.cancel()
        self._is_cancelled = True

    def set_status(self, status):
        self._current_control = self._controls.get(status, None)
        self._status = status
        if self._is_cancelled:
            raise CancelledError()

    def get_status(self):
        if self._current_control is not None:
            return f'{self._status}:  {self._current_control.get_status()}'
        return self._status

    def get_progress(self):
        if self._current_control is not None:
            return self._current_control.get_progress()
        return 0

class Clients(BaseModel):
    gee: GoogleEarthEngine = None

clients = Clients()

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Running pretasks..")
    #Load environment variables
    load_dotenv(find_dotenv())
    # Load the ML model
    clients.gee = GoogleEarthEngine()

    yield
    # Clean up


def get_app():
    app = FastAPI(
        root_path="/v1",
        lifespan=lifespan
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


@dataclasses.dataclass
class InternalState:
    control: Optional[Control]
    current_data: dict
    model_path: Optional[str] = None


internal_state = InternalState(Control({}), {})

state = State(ready=True, status='idle')


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

@app.post('/gee')
async def test_google_earth_engine(file: Union[UploadFile, None] = None, background_tasks: BackgroundTasks = None) -> dict:
    prediction_data = read_zip_folder(file.file)
    prediction_data.climate_data = clients.gee.fetch_data_climate_indicator(file.file, prediction_data.health_data.period_range)
    

@app.post('/zip-file')
async def post_zip_file(file: Union[UploadFile, None] = None, background_tasks: BackgroundTasks = None) -> dict:
    '''
    Post a zip file containing the data needed for training and evaluation, and start the training
    '''
    if not state.ready:
        raise HTTPException(status_code=400, detail="Model is currently training")
    state.ready = False
    state.status = 'training'
    prediction_data = read_zip_folder(file.file)
    model_name, model_path = 'HierarchicalStateModelD', None
    if internal_state.model_path is not None:
        model_name = 'external'
        model_path = internal_state.model_path

    def train_func():
        internal_state.control = Control({'Training': TrainingControl()})
        try:
            internal_state.current_data['response'] = train_on_prediction_data(
                prediction_data,
                model_name=model_name,
                model_path=model_path,
                control=internal_state.control)
        except CancelledError:
            state.status = 'cancelled'
            state.ready = True
            internal_state.control = None
            return
        state.ready = True
        state.status = 'idle'

    background_tasks.add_task(train_func)
    return {'status': 'success'}


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

    if 'response' not in internal_state.current_data:
        raise HTTPException(status_code=400, detail="No response available")
    return internal_state.current_data['response']


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
    if not state.ready:
        state.progress = internal_state.control.get_progress()
        state.status = internal_state.control.get_status()
    return state


def main_backend():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

