import dataclasses
import logging
import time
from enum import Enum
from http.client import HTTPException
from typing import List, Union

from fastapi import BackgroundTasks, UploadFile
from pydantic import BaseModel
# from fastapi.responses import HTMLResponse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from climate_health.api import read_zip_folder, dhis_zip_flow
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
        'https://chess-state-front.vercel.app/'
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
    status: str


state = State(status='idle')


@app.post('/post_data/{data_type}')
async def post_data(data_type: str, rows: List[List[str]]) -> dict:
    current_data[data_type] = parse_json_rows(data_type, rows)
    return {'status': 'success'}


@app.post('/post_zip_file/')
async def post_zip_file(file: Union[UploadFile, None] = None) -> List[dict]:
    out_json = dhis_zip_flow(file.file, model_name='HierarchicalStateModelD')
    # train_data = read_zip_folder(file.file)
    # print(train_data)
    print(out_json)
    return out_json
    # return {'status': 'success'}


@app.post('/train/{model_name}')
async def train_model(model_name: str) -> dict:
    model = get_model(model_name)()
    state.status = 'training'
    return {'status': 'success'}


def main_backend():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
