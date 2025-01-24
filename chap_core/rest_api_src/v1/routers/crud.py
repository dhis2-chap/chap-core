"""
This module contains rest api endpoints for CRUDish operations on the database
Create/Post endpoints will either return the database id of the created object or a job id
that can later be used to retrieve the database id of the created object.

List endpoints will return a list of objects in the database without full data

Get endpoints will return a single object with full data

We try to make the returned objects look as much as possible like the objects in the database
This is achieved by subclassing common basemodels in the read objects and database table objects

Magic is used to make the returned objects camelCase while internal objects are snake_case

"""

import json
from datetime import datetime
from functools import partial
from fastapi import Path
from typing import Optional, List, Annotated

import pandas as pd
from pydantic import BaseModel, Field
from sqlmodel import select

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from sqlmodel import Session

from chap_core.api_types import FeatureCollectionModel
from chap_core.database.database import SessionWrapper
from chap_core.database.model_spec_tables import FeatureSource, ModelSpecRead, ModelSpec
from chap_core.datatypes import FullData, HealthPopulationData
from chap_core.geometry import Polygons
from chap_core.spatio_temporal_data.converters import observations_to_dataset
from .dependencies import get_session, get_database_url, get_settings
from chap_core.rest_api_src.celery_tasks import CeleryPool
from chap_core.database.tables import BackTest, BackTestMetric, BackTestForecast, BackTestBase, Prediction, \
    PredictionRead
from chap_core.database.debug import DebugEntry
from chap_core.database.dataset_tables import ObservationBase, DataSetBase, DataSet, DataSetWithObservations
from chap_core.database.base_tables import DBModel
from chap_core.data import DataSet as InMemoryDataSet
import chap_core.rest_api_src.db_worker_functions as wf

router = APIRouter(prefix="/crud", tags=["crud"])

router_get = partial(router.get, response_model_by_alias=True)  # MAGIC!: This makes the endpoints return camelCase
worker = CeleryPool()


# TODO camel in paths


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


@router.post("/predictions", response_model=JobResponse)
async def create_prediction(prediction: PredictionCreate):
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/predictions/{predictionId}", response_model=PredictionRead)
async def get_prediction(prediction_id: Annotated[int, Path(alias="predictionId")],
                         session: Session = Depends(get_session)):
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return prediction


@router.get("/backtest", response_model=list[BackTestRead])
async def get_backtests(session: Session = Depends(get_session)):
    backtests = session.exec(select(BackTest)).all()
    return backtests


class DatasetCreate(DataSetBase):
    observations: List[ObservationBase]


@router.get('/datasets/{datasetId}', response_model=DataSetWithObservations)
async def get_dataset(dataset_id: Annotated[int, Path(alias='datasetId')], session: Session = Depends(get_session)):
    # dataset = session.exec(select(DataSet).where(DataSet.id == dataset_id)).first()
    dataset = session.get(DataSet, dataset_id)
    assert len(dataset.observations) > 0
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


class DataBaseResponse(DBModel):
    id: int


@router.post('/datasets')
async def create_dataset(data: DatasetCreate, datababase_url=Depends(get_database_url),
                         worker_settings=Depends(get_settings)) -> JobResponse:
    health_data = observations_to_dataset(HealthPopulationData, data.observations, fill_missing=True)
    health_data.set_polygons(FeatureCollectionModel.model_validate_json(data.geojson))
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
async def get_debug_entry(debug_id: Annotated[int, Path(alias='debugId')],
                          session: Session = Depends(get_session)) -> DebugEntry:
    debug = session.get(DebugEntry, debug_id)
    if debug is None:
        raise HTTPException(status_code=404, detail="Debug entry not found")
    return debug


@router.get('/feature-sources', response_model=list[FeatureSource])
def list_feature_types(session: Session = Depends(get_session)):
    return SessionWrapper(session=session).list_all(FeatureSource)


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


@router.get('/models', response_model=list[ModelSpecRead])
def list_models(session: Session = Depends(get_session)):
    return SessionWrapper(session=session).list_all(ModelSpec)
