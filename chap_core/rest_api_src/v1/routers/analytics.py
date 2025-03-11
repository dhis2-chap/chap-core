from typing import List, Annotated

import chap_core.rest_api_src.db_worker_functions as wf
import numpy as np
from pydantic import BaseModel, confloat

from fastapi import APIRouter, HTTPException, Depends, Query, Path
from sqlmodel import Session

from chap_core.api_types import EvaluationEntry, DataList, DataElement, PredictionEntry, FeatureCollectionModel
from chap_core.database.base_tables import DBModel
from chap_core.datatypes import create_tsdataclass
from chap_core.spatio_temporal_data.converters import observations_to_dataset
from .crud import JobResponse
from .dependencies import get_session, get_database_url, get_settings
from chap_core.database.tables import BackTest
from chap_core.database.dataset_tables import DataSet
import logging

from ...celery_tasks import CeleryPool
from ...data_models import DatasetMakeRequest

router = APIRouter(prefix="/analytics", tags=["analytics"])

logger = logging.getLogger(__name__)
worker = CeleryPool()


class EvaluationEntryRequest(BaseModel):
    backtest_id: int
    quantiles: List[confloat(ge=0, le=1)]


class MetaDataEntry(BaseModel):
    element_id: str
    element_name: str
    feature_name: str


class MetaData(BaseModel):
    data_name_mapping: List[MetaDataEntry]


@router.post("/make-dataset", response_model=JobResponse)
def make_dataset(request: DatasetMakeRequest,
                 database_url: str = Depends(get_database_url),
                 worker_settings=Depends(get_settings)):
    """
    This endpoint creates a dataset from the provided data and the data to be fetched3
    and puts it in the database
    """
    feature_names = list({entry.element_id for entry in request.provided_data})
    dataclass = create_tsdataclass(feature_names)
    provided_data = observations_to_dataset(dataclass, request.provided_data, fill_missing=True)
    # provided_field_names = {entry.element_id: entry.element_name for entry in request.provided_data}
    provided_data.set_polygons(FeatureCollectionModel.model_validate(request.geojson))
    job = worker.queue_db(wf.harmonize_and_add_dataset,
                          feature_names,
                          request.data_to_be_fetched,
                          provided_data.model_dump(),
                          request.name,
                          database_url=database_url,
                          worker_config=worker_settings)
    return JobResponse(id=job.id)


@router.get("/evaluation-entry", response_model=List[EvaluationEntry])
async def get_evaluation_entries(
        backtest_id: Annotated[int, Query(alias="backtestId")],
        quantiles: List[float] = Query(...),
        session: Session = Depends(get_session)):
    backtest = session.get(BackTest, backtest_id)
    if backtest is None:
        raise HTTPException(status_code=404, detail="BackTest not found")
    return [
        EvaluationEntry(period=forecast.period, orgUnit=forecast.org_unit, quantile=q,
                        splitPeriod=forecast.last_seen_period,
                        value=np.quantile(forecast.values, q)) for forecast in backtest.forecasts for q in
        quantiles
    ]


class MakePredictionRequest(DatasetMakeRequest):
    model_id: str


class MultiBacktestCreate(DBModel):
    model_ids: List[str]
    dataset_id: int


@router.post("/create_backtests", response_model=List[JobResponse])
async def create_backtest(backtests: MultiBacktestCreate, database_url: str = Depends(get_database_url)):
    job_ids = []
    for model_id in backtests.model_ids:
        job = worker.queue_db(wf.run_backtest, model_id, backtests.dataset_id, 12, 2, 1, database_url=database_url)
        job_ids.append(job.id)

    return [JobResponse(id=job_id) for job_id in job_ids]


@router.post('/prediction', response_model=JobResponse)
async def make_prediction(request: MakePredictionRequest,
                          database_url=Depends(get_database_url),
                          worker_settings=Depends(get_settings)):
    feature_names = list({entry.element_id for entry in request.provided_data})
    dataclass = create_tsdataclass(feature_names)
    provided_data = observations_to_dataset(dataclass, request.provided_data, fill_missing=True)
    # provided_field_names = {entry.element_id: entry.element_name for entry in request.provided_data}
    provided_data.set_polygons(FeatureCollectionModel.model_validate(request.geojson))
    job = worker.queue_db(wf.predict_pipeline_from_composite_dataset,
                          feature_names,
                          request.data_to_be_fetched,
                          provided_data.model_dump(),
                          request.name,
                          request.model_id,
                          database_url=database_url,
                          worker_config=worker_settings)
    return JobResponse(id=job.id)
    # return JobResponse(id=job.id)
    #
    # model_id = dataset.model_id
    # data = dataset_model_to_dataset(HealthPopulationData, dataset)
    # job = worker.queue_db(wf.predict_pipeline_from_health_dataset,
    #                       data.model_dump(), dataset.name, model_id,
    #                       database_url=datababase_url, worker_config=worker_settings)
    #
    # return JobResponse(id=job.id)


@router.get("/prediction-entry/{predictionId}", response_model=List[PredictionEntry])
def get_prediction_entries(prediction_id: Annotated[int, Path(alias="predictionId")],
                           session: Session = Depends(get_session)):
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/actual-cases/{backtestId}", response_model=DataList)
async def get_actual_cases(backtest_id: Annotated[int, Path(alias="backtestId")],
                           session: Session = Depends(get_session)):
    backtest = session.get(BackTest, backtest_id)
    logger.info(f"Backtest: {backtest}")
    data = session.get(DataSet, backtest.dataset_id)
    logger.info(f"Data: {data}")
    if backtest is None:
        raise HTTPException(status_code=404, detail="BackTest not found")
    data_list = [DataElement(pe=observation.period, ou=observation.org_unit, value=float(observation.value) if not (
            np.isnan(observation.value) or observation.value is None) else None) for
                 observation in data.observations if observation.element_id == "disease_cases"]
    logger.info(f"DataList: {len(data_list)}")
    return DataList(
        featureId="disease_cases",
        dhis2Id="disease_cases",
        data=data_list
    )


class DataSource(DBModel):
    name: str
    display_name: str
    supported_features: List[str]
    description: str
    dataset: str


data_sources = [
    DataSource(name='mean_2m_air_temperature',
               display_name='Mean 2m Air Temperature',
               supported_features=['mean_temperature'],
               description='Average air temperature at 2m height (daily average)',
               dataset='era5'),
    DataSource(name='minimum_2m_air_temperature',
               display_name='Minimum 2m Air Temperature',
               supported_features=[''],
               description='Minimum air temperature at 2m height (daily minimum)',
               dataset='era5'),
    DataSource(name='maximum_2m_air_temperature',
               display_name='Maximum 2m Air Temperature',
               supported_features=[''],
               description='Maximum air temperature at 2m height (daily maximum)',
               dataset='era5'),
    DataSource(name='dewpoint_2m_temperature',
               display_name='Dewpoint 2m Temperature',
               supported_features=[''],
               description='Dewpoint temperature at 2m height (daily average)',
               dataset='era5'),
    DataSource(name='total_precipitation',
               display_name='Total Precipitation',
               supported_features=['rainfall'],
               description='Total precipitation (daily sums)',
               dataset='era5'),
    DataSource(name='surface_pressure',
               display_name='Surface Pressure',
               supported_features=['surface_pressure'],
               description='Surface pressure (daily average)',
               dataset='era5'),
    DataSource(name='mean_sea_level_pressure',
               display_name='Mean Sea Level Pressure',
               supported_features=['mean_sea_level_pressure'],
               description='Mean sea level pressure (daily average)',
               dataset='era5'),
    DataSource(name='u_component_of_wind_10m',
               display_name='U Component of Wind 10m',
               supported_features=['u_component_of_wind_10m'],
               description='10m u-component of wind (daily average)',
               dataset='era5'),
    DataSource(name='v_component_of_wind_10m',
               display_name='V Component of Wind 10m',
               supported_features=['v_component_of_wind_10m'],
               description='10m v-component of wind (daily average)',
               dataset='era5'),
]


@router.get('/data-sources')
async def get_data_sources() -> list[DataSource]:
    return data_sources
