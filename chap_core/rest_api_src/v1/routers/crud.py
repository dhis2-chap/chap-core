import json
from datetime import datetime
from functools import partial
from fastapi import Path
from typing import Optional, List, Annotated

import pandas as pd
from pydantic import BaseModel, Field
from sqlmodel import select

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Query
from sqlmodel import Session

from chap_core.api_types import FeatureCollectionModel, DataListV2
from chap_core.database.database import SessionWrapper
from chap_core.datatypes import FullData
from chap_core.geometry import Polygons
from .dependencies import get_session, get_database_url, get_settings
from chap_core.rest_api_src.celery_tasks import CeleryPool
from chap_core.database.tables import BackTest, DataSet, BackTestMetric, BackTestForecast, DebugEntry, \
    DataSetWithObservations, DBModel, BackTestBase
from chap_core.data import DataSet as InMemoryDataSet
import chap_core.rest_api_src.db_worker_functions as wf
import chap_core.rest_api_src.worker_functions as normal_wf

router = APIRouter(prefix="/crud", tags=["crud"])

router_get = partial(router.get, response_model_by_alias=True)  # MAGIC!: This makes the endpoints return camelCase
worker = CeleryPool()


class JobResponse(BaseModel):
    id: str


class BackTestCreate(BackTestBase):
    ...


class BackTestRead(BackTestBase):
    id: int
    name: Optional[str] = None
    timestamp: Optional[datetime] = None

    # THis is dataset properties
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    org_unit_ids: List[str] = Field(default_factory=list)


class BackTestFull(BackTestRead):
    metrics: list[BackTestMetric]
    forecasts: list[BackTestForecast]


@router_get("/backtest/{backtestId}", response_model=BackTestFull)
async def get_backtest(backtest_id: Annotated[int, Path(alias="backtestId")],
                       session: Session = Depends(get_session)):
    backtest = session.get(BackTest, backtest_id)
    if backtest is None:
        raise HTTPException(status_code=404, detail="BackTest not found")
    return backtest


@router.post("/backtest", response_model=JobResponse)
async def create_backtest(backtest: BackTestCreate, database_url: str = Depends(get_database_url)):
    job = worker.queue_db(wf.run_backtest, backtest.model_id, backtest.dataset_id, 12, 2, 1,
                          database_url=database_url)
    return JobResponse(id=job.id)


class PredictionCreate(DBModel):
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


class DatasetCreate(DBModel):
    name: str
    orgUnitsGeoJson: FeatureCollectionModel
    features: list[DataListV2]


@router.get('/datasets/{datasetId}', response_model=DataSetWithObservations)
async def get_dataset(dataset_id: Annotated[int, Path(alias='datasetId')], session: Session = Depends(get_session)):
    dataset = session.exec(select(DataSet).where(DataSet.id == dataset_id)).first()
    assert len(dataset.observations) > 0
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset
    # return DataSetFull(id=dataset.id, name=dataset.name, polygons=dataset.polygons, observations=dataset.observations)


class DataBaseResponse(DBModel):
    id: int


@router.post('/datasets/json')
async def create_dataset(data: DatasetCreate, datababase_url=Depends(get_database_url),
                         worker_settings=Depends(get_settings)) -> JobResponse:
    health_data = normal_wf.get_health_dataset(data, colnames=['orgUnit', 'period'])
    job = worker.queue_db(wf.harmonize_and_add_health_dataset, health_data.model_dump(), data.name,
                          database_url=datababase_url, worker_config=worker_settings)
    return JobResponse(id=job.id)


@router.post('/datasets/csvFile')
async def create_dataset_csv(csv_file: UploadFile = File(..., alias='csvFile'),
                             geojson_file: UploadFile = File(..., alias='geojsonFile'),
                             session: Session = Depends(get_session),
                             ) -> DataBaseResponse:
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


@router.get('/debug/{debugId}')
async def get_debug_entry(debug_id: Annotated[int, Path(alias='debugId')], session: Session = Depends(get_session)) -> DebugEntry:
    debug = session.get(DebugEntry, debug_id)
    if debug is None:
        raise HTTPException(status_code=404, detail="Debug entry not found")
    return debug


class DataBaseResponse(DBModel):
    id: int


class DataSetRead(DBModel):
    id: int
    name: str

    class Config:
        orm_mode = True  # Enable compatibility with ORM models


@router.get('/datasets', response_model=list[DataSetRead])
async def get_datasets(session: Session = Depends(get_session)):
    datasets = session.exec(select(DataSet)).all()
    return datasets
