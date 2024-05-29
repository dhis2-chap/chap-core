import dataclasses
import logging
import time
from enum import Enum
from typing import List, Union

from fastapi import BackgroundTasks, UploadFile, HTTPException
from pydantic import BaseModel
# from fastapi.responses import HTMLResponse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from climate_health.api import read_zip_folder, dhis_zip_flow, train_on_prediction_data
from climate_health.dhis2_interface.json_parsing import parse_json_rows
from climate_health.predictor import ModelType, get_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_app():
    app = FastAPI()
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

current_data = {}


class State(BaseModel):
    ready: bool
    status: str


state = State(ready=True, status='idle')


def set_cur_response(response):
    state['response'] = response


class PredictionResponse(BaseModel):
    value: float
    orgUnit: str
    dataElement: str
    period: str


@app.post('/post_zip_file/')
async def post_zip_file(file: Union[UploadFile, None] = None, background_tasks: BackgroundTasks = None) -> dict:
    '''
    Post a zip file containing the data needed for training and evaluation, and start the training
    '''
    if not state.ready:
        raise HTTPException(status_code=400, detail="Model is currently training")
    state.ready = False
    state.status = 'training'
    prediction_data = read_zip_folder(file.file)

    def train_func():
        current_data['response'] = train_on_prediction_data(prediction_data, model_name='HierarchicalStateModelD')
        state.ready = True
        state.status = 'idle'

    background_tasks.add_task(train_func)
    return {'status': 'success'}


@app.get('/get_results/')
async def get_results() -> List[PredictionResponse]:
    '''
    Retrieve results made by the model
    '''

    if 'response' not in current_data:
        raise HTTPException(status_code=400, detail="No response available")
    return current_data['response']


@app.get('/status/')
async def get_status() -> State:
    '''
    Retrieve the current status of the model
    '''
    return state


def main_backend():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
