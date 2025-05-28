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
import logging

import numpy as np
from fastapi import Path, Query
from typing import Optional, List, Annotated

import pandas as pd
from sqlmodel import select

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from sqlmodel import Session

from chap_core.api_types import FeatureCollectionModel
from chap_core.database.database import SessionWrapper
from chap_core.database.model_spec_tables import ModelSpecRead
from chap_core.datatypes import FullData, HealthPopulationData
from chap_core.geometry import Polygons
from chap_core.spatio_temporal_data.converters import observations_to_dataset
from .dependencies import get_session, get_database_url, get_settings
from chap_core.rest_api_src.celery_tasks import CeleryPool
from chap_core.database.tables import BackTest, Prediction, PredictionRead, PredictionInfo
from chap_core.database.debug import DebugEntry
from chap_core.database.dataset_tables import ObservationBase, DataSetBase, DataSet, DataSetWithObservations
from chap_core.database.model_templates_and_config_tables import (
    ConfiguredModelDB,
    ModelTemplateDB,
    ModelTemplateMetaData,
    ModelTemplateInformation,
    ModelConfiguration,
)
from chap_core.database.base_tables import DBModel
from chap_core.data import DataSet as InMemoryDataSet
import chap_core.rest_api_src.db_worker_functions as wf
from ...data_models import JobResponse, BackTestCreate, BackTestRead, BackTestFull

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/crud", tags=["crud"])

router_get = partial(router.get, response_model_by_alias=True)  # MAGIC!: This makes the endpoints return camelCase
worker = CeleryPool()


@router.get("/backtests", response_model=list[BackTestRead])  # This should be called list
async def get_backtests(session: Session = Depends(get_session)):
    """
    Returns a list of backtests/evaluations with only the id and name
    """
    backtests = session.exec(select(BackTest)).all()
    return backtests


@router_get("/backtests/{backtestId}", response_model=BackTestFull)
async def get_backtest(backtest_id: Annotated[int, Path(alias="backtestId")], session: Session = Depends(get_session)):
    backtest = session.get(BackTest, backtest_id)
    if backtest is None:
        raise HTTPException(status_code=404, detail="BackTest not found")
    return backtest


class BackTestUpdate(DBModel):
    name: str = None


@router.post("/backtests", response_model=JobResponse)
async def create_backtest(backtest: BackTestCreate, database_url: str = Depends(get_database_url)):
    job = worker.queue_db(wf.run_backtest, backtest, database_url=database_url)

    return JobResponse(id=job.id)


@router.delete("/backtests/{backtestId}")
async def delete_backtest(
    backtest_id: Annotated[int, Path(alias="backtestId")], session: Session = Depends(get_session)
):
    backtest = session.get(BackTest, backtest_id)
    if backtest is None:
        raise HTTPException(status_code=404, detail="BackTest not found")
    session.delete(backtest)
    session.commit()
    return {"message": "deleted"}


@router.patch("/backtests/{backtestId}", response_model=BackTestRead)
async def update_backtest(
    backtest_id: Annotated[int, Path(alias="backtestId")],
    backtest_update: BackTestUpdate,
    session: Session = Depends(get_session),
):
    db_backtest = session.get(BackTest, backtest_id)
    if not db_backtest:
        raise HTTPException(status_code=404, detail="BackTest not found")

    update_data = backtest_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_backtest, key, value)

    session.add(db_backtest)
    session.commit()
    session.refresh(db_backtest)
    return db_backtest


@router.delete("/backtests")
async def delete_backtest_batch(ids: Annotated[str, Query(alias="ids")], session: Session = Depends(get_session)):
    deleted_count = 0
    backtest_ids_list = []

    if not ids:
        raise HTTPException(status_code=400, detail="No backtest IDs provided.")
    raw_id_parts = ids.split(",")
    if not any(part.strip() for part in raw_id_parts):
        raise HTTPException(
            status_code=400, detail="No valid IDs provided. Input consists of only commas or whitespace."
        )
    for id_str_part in raw_id_parts:
        stripped_id_str = id_str_part.strip()
        if not stripped_id_str:
            # Handle empty segments from inputs like "1,,2" or "1,"
            raise HTTPException(
                status_code=400,
                detail=f"Invalid ID format: found empty ID segment in '{ids}'. IDs must be non-empty, comma-separated integers.",
            )
        try:
            backtest_ids_list.append(int(stripped_id_str))
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid ID format: '{stripped_id_str}' is not a valid integer in '{ids}'."
            )

    for backtest_id in backtest_ids_list:
        backtest = session.get(BackTest, backtest_id)
        if backtest is not None:
            session.delete(backtest)
            deleted_count += 1
    session.commit()
    return {"message": f"Deleted {deleted_count} backtests"}


class PredictionCreate(DBModel):
    dataset_id: int
    estimator_id: str
    n_periods: int


@router.get("/predictions", response_model=list[PredictionInfo])
async def get_predictions(session: Session = Depends(get_session)):
    session_wrapper = SessionWrapper(session=session)
    return session_wrapper.list_all(Prediction)


@router.get("/predictions/{predictionId}", response_model=PredictionRead)
async def get_prediction(
    prediction_id: Annotated[int, Path(alias="predictionId")], session: Session = Depends(get_session)
):
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return prediction


@router.post("/predictions", response_model=JobResponse)
async def create_prediction(prediction: PredictionCreate):
    raise HTTPException(status_code=501, detail="Not implemented")


@router.delete("/predictions/{predictionId}")
async def delete_prediction(
    prediction_id: Annotated[int, Path(alias="predictionId")], session: Session = Depends(get_session)
):
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    session.delete(prediction)
    session.commit()
    return {"message": "deleted"}


###########
# datasets


class DataBaseResponse(DBModel):
    id: int


class DatasetCreate(DataSetBase):
    observations: List[ObservationBase]
    geojson: FeatureCollectionModel


class DataSetRead(DBModel):
    id: int
    name: str
    type: Optional[str]
    created: Optional[datetime]
    covariates: List[str]


@router.get("/datasets", response_model=list[DataSetRead])
async def get_datasets(session: Session = Depends(get_session)):
    datasets = session.exec(select(DataSet)).all()
    return datasets


@router.get("/datasets/{datasetId}", response_model=DataSetWithObservations)
async def get_dataset(dataset_id: Annotated[int, Path(alias="datasetId")], session: Session = Depends(get_session)):
    # dataset = session.exec(select(DataSet).where(DataSet.id == dataset_id)).first()
    dataset = session.get(DataSet, dataset_id)
    assert len(dataset.observations) > 0
    for obs in dataset.observations:
        try:
            obs.value = obs.value if obs.value is None or np.isfinite(obs.value) else None
        except:
            raise
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@router.post("/datasets")
async def create_dataset(
    data: DatasetCreate, datababase_url=Depends(get_database_url), worker_settings=Depends(get_settings)
) -> JobResponse:
    health_data = observations_to_dataset(HealthPopulationData, data.observations, fill_missing=True)
    health_data.set_polygons(FeatureCollectionModel.model_validate(data.geojson))
    job = worker.queue_db(
        wf.harmonize_and_add_health_dataset,
        health_data.model_dump(),
        data.name,
        database_url=datababase_url,
        worker_config=worker_settings,
    )
    return JobResponse(id=job.id)


@router.post("/datasets/csvFile")
async def create_dataset_csv(
    csv_file: UploadFile = File(...),
    geojson_file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> DataBaseResponse:
    csv_content = await csv_file.read()
    dataset = InMemoryDataSet.from_csv(pd.io.common.BytesIO(csv_content), dataclass=FullData)
    geo_json_content = await geojson_file.read()
    features = Polygons.from_geojson(json.loads(geo_json_content), id_property="NAME_1").feature_collection()
    dataset_id = SessionWrapper(session=session).add_dataset("csv_file", dataset, features.model_dump_json())
    return DataBaseResponse(id=dataset_id)


@router.delete("/datasets/{datasetId}")
async def delete_dataset(dataset_id: Annotated[int, Path(alias="datasetId")], session: Session = Depends(get_session)):
    # dataset = session.exec(select(DataSet).where(DataSet.id == dataset_id)).first()
    dataset = session.get(DataSet, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    session.delete(dataset)
    session.commit()
    return {"message": "deleted"}


###########
# model templates


class ModelTemplateRead(DBModel, ModelTemplateInformation, ModelTemplateMetaData):
    """
    ModelTemplateRead is a read model for the ModelTemplateDB.
    It is used to return the model template in a readable format.
    """

    # TODO: should probably be moved somewhere else?
    name: str
    id: int
    user_options: Optional[dict] = None
    required_covariates: List[str] = []


@router.get("/model-templates", response_model=list[ModelTemplateRead])
async def list_model_templates(session: Session = Depends(get_session)):
    """
    Lists all model templates from the db.
    """
    model_templates = session.exec(select(ModelTemplateDB)).all()
    return model_templates


###########
# configured models


@router.get("/configured-models", response_model=list[ModelSpecRead])
def list_configured_models(session: Session = Depends(get_session)):
    """List all configured models from the db"""
    configured_models_read = SessionWrapper(session=session).get_configured_models()

    # return
    return configured_models_read


class ModelConfigurationCreate(DBModel):
    name: str
    model_template_id: int
    user_option_values: Optional[dict] = None
    additional_continuous_covariates: List[str] = []


@router.post("/configured-models", response_model=ConfiguredModelDB)
def add_configured_model(
    model_configuration: ModelConfigurationCreate,
    session: SessionWrapper = Depends(get_session),  # type: ignore[call-arg]
):
    """Add a configured model to the database"""
    session_wrapper = SessionWrapper(session=session)
    model_template_id = model_configuration.model_template_id
    configuration_name = model_configuration.name
    db_id = session_wrapper.add_configured_model(
        model_template_id, ModelConfiguration(**model_configuration.dict()), configuration_name
    )
    return session.get(ConfiguredModelDB, db_id)


###########
# models (alias for configured models)


@router.get("/models", response_model=list[ModelSpecRead])
def list_models(session: Session = Depends(get_session)):
    """List all models from the db (alias for configured models)"""
    return list_configured_models(session)


@router.post("/models", response_model=ConfiguredModelDB)
def add_model(
    model_configuration: ModelConfigurationCreate,
    session: SessionWrapper = Depends(get_session),  # type: ignore[call-arg]
):
    """Add a model to the database (alias for configured models)"""
    return add_configured_model(model_configuration, session)


#############
# other misc


@router.post("/debug")
async def debug_entry(database_url: str = Depends(get_database_url)) -> JobResponse:
    job = worker.queue_db(wf.debug, database_url=database_url)
    return JobResponse(id=job.id)


@router.get("/debug/{debugId}")
async def get_debug_entry(
    debug_id: Annotated[int, Path(alias="debugId")], session: Session = Depends(get_session)
) -> DebugEntry:
    debug = session.get(DebugEntry, debug_id)
    if debug is None:
        raise HTTPException(status_code=404, detail="Debug entry not found")
    return debug
