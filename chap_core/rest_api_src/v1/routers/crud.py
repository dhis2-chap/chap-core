import json
from datetime import datetime
from typing import Optional, List

import pandas as pd
from pydantic import BaseModel, Field
from sqlmodel import select

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from sqlmodel import Session

from chap_core.api_types import RequestV1
from chap_core.database.database import SessionWrapper
from chap_core.datatypes import FullData
from chap_core.geometry import Polygons
from .dependencies import get_session, get_database_url
from chap_core.rest_api_src.celery_tasks import CeleryPool
from chap_core.database.tables import BackTest, DataSet, BackTestMetric, BackTestForecast, DebugEntry
from chap_core.data import DataSet as InMemoryDataSet
import chap_core.rest_api_src.db_worker_functions as wf
import chap_core.rest_api_src.worker_functions as normal_wf

router = APIRouter(prefix="/crud", tags=["crud"])
worker = CeleryPool()


class JobResponse(BaseModel):
    id: str


class BackTestCreate(BaseModel):
    dataset_id: int
    estimator_id: str


class BackTestRead(BackTestCreate):
    id: int
    name: Optional[str] = None
    timestamp: Optional[datetime] = None

    # THis is dataset properties
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    org_unit_ids: List[str] = Field(default_factory=list)

    class Config:
        orm_mode = True  # Enable compatibility with ORM models


class BackTestFull(BackTestRead):
    metrics: list[BackTestMetric]
    forecasts: list[BackTestForecast]


@router.get("/backtest/{backtest_id}", response_model=BackTestFull)
async def get_backtest(backtest_id: int, session: Session = Depends(get_session)):
    backtest = session.get(BackTest, backtest_id)
    if backtest is None:
        raise HTTPException(status_code=404, detail="BackTest not found")
    return backtest


@router.post("/backtest", response_model=JobResponse)
async def create_backtest(backtest: BackTestCreate, database_url: str = Depends(get_database_url)):
    job = worker.queue_db(wf.run_backtest, backtest.estimator_id, backtest.dataset_id, 12, 2, 1, database_url=database_url)
    return JobResponse(id=job.id)


class PredictionCreate(BaseModel):
    dataset_id: int
    estimator_id: str
    n_periods: int


@router.post("/prediction", response_model=JobResponse)
async def create_prediction(prediction: PredictionCreate):
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/backtest", response_model=list[BackTestRead])
async def get_backtests(session: Session = Depends(get_session)):
    backtests = session.exec(select(BackTest)).all()
    return backtests


@router.get('/dataset/{dataset_id}', response_model=DataSet)
async def get_dataset(dataset_id: int, session: Session = Depends(get_session)):
    dataset = session.get(DataSet, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


class DatasetCreate(RequestV1):
    name: str


class DataBaseResponse(BaseModel):
    id: int


@router.post('/dataset/json')
async def create_dataset(data: DatasetCreate, datababase_url=Depends(get_database_url)) -> JobResponse:
    health_data = normal_wf.get_health_dataset(data)
    job = worker.queue_db(wf.harmonize_and_add_health_dataset, health_data.model_dump(), data.name, database_url=datababase_url)
    return JobResponse(id=job.id)


@router.post('/dataset/csv_file')
async def create_dataset_csv(csv_file: UploadFile = File(...),
                             geojson_file: UploadFile = File(...),
                             session: Session = Depends(get_session)) -> DataBaseResponse:
    csv_content = await csv_file.read()
    dataset = InMemoryDataSet.from_csv(pd.io.common.BytesIO(csv_content), dataclass=FullData)
    geo_json_content = await geojson_file.read()
    features = Polygons.from_geojson(json.loads(geo_json_content), id_property='NAME_1').feature_collection()
    dataset_id = SessionWrapper(session=session).add_dataset('csv_file', dataset, features.model_dump_json())
    return DataBaseResponse(id=dataset_id)


@router.post('/debug')
async def debug_entry(database_url: str = Depends(get_database_url)) -> JobResponse:
    job = worker.queue_db(wf.debug, database_url=database_url)
    return JobResponse(id=job.id)


@router.get('/debug/{debug_id}')
async def get_debug_entry(debug_id: int, session: Session = Depends(get_session)):
    debug = session.get(DebugEntry, debug_id)
    if debug is None:
        raise HTTPException(status_code=404, detail="Debug entry not found")
    return debug


class DataBaseResponse(BaseModel):
    id: int


class DataSetRead(BaseModel):
    id: int
    name: str

    class Config:
        orm_mode = True  # Enable compatibility with ORM models


@router.get('/datasets', response_model=list[DataSetRead])
async def get_datasets(session: Session = Depends(get_session)):
    datasets = session.exec(select(DataSet)).all()
    return datasets
