import logging
from typing import Annotated, List

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, confloat
from sqlalchemy.orm import selectinload
from sqlmodel import Session, select

import chap_core.rest_api.db_worker_functions as wf
from chap_core.api_types import (
    DataElement,
    DataList,
    EvaluationEntry,
    FeatureCollectionModel,
    PredictionEntry,
    BackTestParams,
)
from chap_core.database.base_tables import DBModel
from chap_core.database.dataset_tables import Observation, DataSetCreateInfo, DataSet as DataSetTable
from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB
from chap_core.database.tables import BackTest, BackTestForecast, Prediction
from chap_core.datatypes import create_tsdataclass
from chap_core.spatio_temporal_data.converters import observations_to_dataset
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

from ...celery_tasks import JOB_NAME_KW, JOB_TYPE_KW, CeleryPool
from ...data_models import (
    BackTestCreate,
    BackTestRead,
    DatasetMakeRequest,
    ImportSummaryResponse,
    JobResponse,
    PredictionParams,
    ValidationError,
)
from .dependencies import get_database_url, get_session, get_settings

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


@router.post("/make-dataset", response_model=ImportSummaryResponse)
def make_dataset(
    request: DatasetMakeRequest, database_url: str = Depends(get_database_url), worker_settings=Depends(get_settings)
):
    """
    This endpoint creates a dataset from the provided data and the data to be fetched3
    and puts it in the database
    """
    feature_names, provided_data = _read_dataset(request)
    provided_data, rejections = _validate_full_dataset(feature_names, provided_data)
    # provided_field_names = {entry.element_id: entry.element_name for entry in request.provided_data}
    polygon_rejected = provided_data.set_polygons(FeatureCollectionModel.model_validate(request.geojson))
    rejections.extend(
        ValidationError(reason="Missing polygon in geojson", orgUnit=location, feature_name="polygon", time_periods=[])
        for location in polygon_rejected
    )
    imported_count = len(provided_data.locations())
    if imported_count == 0:
        raise HTTPException(status_code=500, detail="Missing values. No data was imported.")
    request.type = "evaluation"

    job = worker.queue_db(
        wf.harmonize_and_add_dataset,
        feature_names,
        request.data_to_be_fetched,
        provided_data.model_dump(),
        request.name,
        request.type,
        database_url=database_url,
        worker_config=worker_settings,
        **{JOB_TYPE_KW: "create_dataset", JOB_NAME_KW: request.name},
    )

    return ImportSummaryResponse(id=job.id, imported_count=imported_count, rejected=rejections)


def _read_dataset(request):
    feature_names = list({entry.feature_name for entry in request.provided_data})
    dataclass = create_tsdataclass(feature_names)
    provided_data = observations_to_dataset(dataclass, request.provided_data, fill_missing=True)
    if "population" in feature_names:
        provided_data = provided_data.interpolate(["population"])
    return feature_names, provided_data


def _validate_full_dataset(
    feature_names, provided_data, target_name="disease_cases"
) -> tuple[DataSet, list[ValidationError]]:
    new_data = {}
    rejected_list = []
    n_locations = len(provided_data.locations())
    for location, data in provided_data.items():
        for feature_name in feature_names:
            if feature_name == target_name:
                continue
            isnan = np.isnan(getattr(data, feature_name))
            if np.any(isnan):
                isnan_ = [data.time_period[i].id for i in np.flatnonzero(isnan)]
                rejected_list.append(
                    ValidationError(
                        reason="Missing value for some/all time periods",
                        orgUnit=location,
                        feature_name=feature_name,
                        time_periods=isnan_,
                    )
                )
                break
        else:
            new_data[location] = data
    if not new_data:
        raise HTTPException(
            status_code=500, detail=f"All regions regjected due to missing values. Rejected: {str(rejected_list)}"
        )
    else:
        print(new_data.keys())

    new_dataset = provided_data.__class__(new_data, polygons=provided_data.polygons, metadata=provided_data.metadata)
    logger.info(
        f"Remaining dataset after validation: {len(new_dataset.locations())} (from {n_locations}) locations: {list(new_dataset.locations())} "
    )
    assert len(new_dataset.locations()), new_dataset.locations()
    return new_dataset, rejected_list


@router.get("/compatible-backtests/{backtestId}", response_model=List[BackTestRead])
def get_compatible_backtests(
    backtest_id: Annotated[int, Path(alias="backtestId")], session: Session = Depends(get_session)
):
    """Return a list of backtests that are compatible for comparison with the given backtest"""
    logger.info(f"Checking compatible backtests for {backtest_id}")
    backtest = session.get(BackTest, backtest_id)
    if backtest is None:
        raise HTTPException(status_code=404, detail="BackTest not found")
    org_units = set(backtest.org_units)
    split_periods = set(backtest.split_periods)
    res = session.exec(
        select(BackTest.id, BackTest.org_units, BackTest.split_periods).where(BackTest.id != backtest_id)
    ).all()
    ids = [id for id, o, s in res if set(o) & org_units and set(s) & split_periods]
    backtests = session.exec(
        select(BackTest)
        .where(BackTest.id.in_(ids))
        .options(
            selectinload(BackTest.dataset).defer(DataSetTable.geojson),
            selectinload(BackTest.configured_model).selectinload(ConfiguredModelDB.model_template),
        )
    ).all()
    return backtests


class BacktestDomain(DBModel):
    org_units: List[str]
    split_periods: List[str]


@router.get("/backtest-overlap/{backtestId1}/{backtestId2}", response_model=BacktestDomain)
def get_backtest_overlap(
    backtest_id1: Annotated[int, Path(alias="backtestId1")],
    backtest_id2: Annotated[int, Path(alias="backtestId2")],
    session: Session = Depends(get_session),
):
    """Return the org units and split periods that are common between two backtests"""
    backtest1 = session.get(BackTest, backtest_id1)
    backtest2 = session.get(BackTest, backtest_id2)
    if backtest1 is None:
        raise HTTPException(status_code=404, detail="BackTest 1 not found")
    if backtest2 is None:
        raise HTTPException(status_code=404, detail="BackTest 2 not found")
    org_units1 = list(set(backtest1.org_units) & set(backtest2.org_units))
    split_periods1 = list(set(backtest1.split_periods) & set(backtest2.split_periods))
    return BacktestDomain(org_units=org_units1, split_periods=split_periods1)


@router.get("/prediction-entry", response_model=List[PredictionEntry])
async def get_prediction_entry(
    prediction_id: Annotated[int, Query(alias="predictionId")],
    quantiles: List[float] = Query(...),
    session: Session = Depends(get_session),
):
    """
    return
    """
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return [
        PredictionEntry(
            period=forecast.period,
            orgUnit=forecast.org_unit,
            quantile=q,
            value=np.quantile(forecast.values, q),
        )
        for forecast in prediction.forecasts
        for q in quantiles
    ]


@router.get("/evaluation-entry", response_model=List[EvaluationEntry])
async def get_evaluation_entries(
    backtest_id: Annotated[int, Query(alias="backtestId")],
    quantiles: List[float] = Query(...),
    split_period: str = Query(None, alias="splitPeriod"),
    org_units: List[str] = Query(None, alias="orgUnits"),
    session: Session = Depends(get_session),
):
    """
    Return quantiles for the forecasts in a backtest. Can optionally be filtered on split period and org units.
    NOTE: If org_units is set to ["adm0"], the sum over all regions is returned.
    """
    return_summed = False
    if org_units is not None and len(org_units) == 1 and org_units[0] == "adm0":
        # returning sum of forecasts for all regions
        return_summed = True

    logger.info(
        f"Backtest ID: {backtest_id}, Quantiles: {quantiles}, Split Period: {split_period}, Org Units: {org_units} "
    )
    if backtest_id is not None:
        backtest = session.get(BackTest, backtest_id)
    if backtest is None:
        raise HTTPException(status_code=404, detail="BackTest not found")
    if org_units is not None:
        org_units = set(org_units)
        logger.info("Filtering evaluation entries to org_units: %s", org_units)

    cls = BackTestForecast
    expr = select(cls).where(cls.backtest_id == backtest_id)
    if split_period:
        expr = expr.where(cls.last_seen_period == split_period)
    if org_units and not return_summed:
        expr = expr.where(cls.org_unit.in_(org_units))
    forecasts = session.exec(expr)

    logger.info(forecasts)

    if return_summed:
        # sum forecasts over all regions
        summed_forecasts = {}
        for forecast in forecasts:
            key = (forecast.period, forecast.last_seen_period)
            if key not in summed_forecasts:
                summed_forecasts[key] = np.array([0.0] * len(forecast.values))
            summed_forecasts[key] += np.array(forecast.values)

        forecasts = [
            BackTestForecast(
                period=key[0],
                org_unit="adm0",
                last_seen_period=key[1],
                values=values.tolist(),
            )
            for key, values in summed_forecasts.items()
        ]

    return [
        EvaluationEntry(
            period=forecast.period,
            orgUnit=forecast.org_unit,
            quantile=q,
            splitPeriod=forecast.last_seen_period,
            value=np.quantile(forecast.values, q),
        )
        for forecast in forecasts
        for q in quantiles
    ]


class MakePredictionRequest(DatasetMakeRequest, PredictionParams):
    meta_data: dict = {}


class MakeBacktestRequest(BackTestParams):
    name: str
    model_id: str
    dataset_id: int


class MakeBacktestWithDataRequest(DatasetMakeRequest, BackTestParams):
    name: str
    model_id: str


@router.post("/create-backtest", response_model=JobResponse)
async def create_backtest(request: MakeBacktestRequest, database_url: str = Depends(get_database_url)):
    job = worker.queue_db(
        wf.run_backtest,
        BackTestCreate(name=request.name, dataset_id=request.dataset_id, model_id=request.model_id),
        request.n_periods,
        request.n_splits,
        request.stride,
        database_url=database_url,
        **{JOB_TYPE_KW: "create_backtest", JOB_NAME_KW: request.name},
    )

    return JobResponse(id=job.id)


@router.post("/make-prediction", response_model=JobResponse)
async def make_prediction(
    request: MakePredictionRequest, database_url=Depends(get_database_url), worker_settings=Depends(get_settings)
):
    request.type = "prediction"
    feature_names = list({entry.feature_name for entry in request.provided_data})
    dataclass = create_tsdataclass(feature_names)
    provided_data = observations_to_dataset(dataclass, request.provided_data, fill_missing=True)
    if "population" in feature_names:
        provided_data = provided_data.interpolate(["population"])
    provided_data, rejections = _validate_full_dataset(feature_names, provided_data)
    provided_data.set_polygons(FeatureCollectionModel.model_validate(request.geojson))
    if request.data_to_be_fetched:
        raise HTTPException(status_code=404, detail="Data to be fetched is no longer supported by chap-core")
    dump = request.model_dump()
    dataset_info = DataSetCreateInfo(**dump).model_dump()
    prediction_params = PredictionParams(**dump)
    job = worker.queue_db(
        wf.predict_pipeline_from_composite_dataset,
        feature_names,
        provided_data.model_dump(),
        request.name,
        dataset_create_info=dataset_info,
        prediction_params=prediction_params,
        database_url=database_url,
        worker_config=worker_settings,
        **{JOB_TYPE_KW: "create_prediction", JOB_NAME_KW: request.name},
    )
    return JobResponse(id=job.id)


@router.get("/prediction-entry/{predictionId}", response_model=List[PredictionEntry])
def get_prediction_entries(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    quantiles: List[float] = Query(...),
    session: Session = Depends(get_session),
):
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return [
        PredictionEntry(
            period=forecast.period, orgUnit=forecast.org_unit, quantile=q, value=np.quantile(forecast.values, q)
        )
        for forecast in prediction.forecasts
        for q in quantiles
    ]
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/actualCases/{backtestId}", response_model=DataList)
async def get_actual_cases(
    backtest_id: Annotated[int, Path(alias="backtestId")],
    org_units: List[str] = Query(None, alias="orgUnits"),
    is_dataset_id: bool = Query(False, alias="isDatasetId"),
    session: Session = Depends(get_session),
):
    """
    Return the actual disease cases corresponding to a backtest. Can optionally be filtered on org units.

    Note: If org_units is set to ["adm0"], the sum over all regions is returned.
    """
    return_summed = False
    if org_units is not None and len(org_units) == 1 and org_units[0] == "adm0":
        # returning sum of forecasts for all regions
        return_summed = True
    if not is_dataset_id:
        backtest = session.get(BackTest, backtest_id)
        logger.info(f"Backtest: {backtest}")
        if backtest is None:
            raise HTTPException(status_code=404, detail="BackTest not found")
        dataset_id = backtest.dataset_id
    else:
        dataset_id = backtest_id
    expr = select(Observation).where(Observation.dataset_id == dataset_id)
    if org_units is not None and not return_summed:
        org_units = set(org_units)
        expr = expr.where(Observation.org_unit.in_(org_units))
    observations = session.exec(expr).all()
    logger.info(f"Observations: {observations}")
    data_list = [
        DataElement(
            pe=observation.period,
            ou=observation.org_unit,
            value=float(observation.value)
            if not (observation.value is None or np.isnan(observation.value) or observation.value is None)
            else None,
        )
        for observation in observations
        if observation.feature_name == "disease_cases"
    ]
    if return_summed:
        # sum over all regions
        summed_values = {}
        for element in data_list:
            key = element.pe
            if key not in summed_values:
                summed_values[key] = 0.0
            if element.value is not None:
                summed_values[key] += element.value
        data_list = [DataElement(pe=pe, ou="adm0", value=value) for pe, value in summed_values.items()]
    logger.info(f"DataList: {len(data_list)}")
    return DataList(featureId="disease_cases", dhis2Id="disease_cases", data=data_list)


class ChapDataSource(DBModel):
    name: str
    display_name: str
    supported_features: List[str]
    description: str
    dataset: str


data_sources = [
    ChapDataSource(
        name="mean_2m_air_temperature",
        display_name="Mean 2m Air Temperature",
        supported_features=["mean_temperature"],
        description="Average air temperature at 2m height (daily average)",
        dataset="era5",
    ),
    ChapDataSource(
        name="minimum_2m_air_temperature",
        display_name="Minimum 2m Air Temperature",
        supported_features=[""],
        description="Minimum air temperature at 2m height (daily minimum)",
        dataset="era5",
    ),
    ChapDataSource(
        name="maximum_2m_air_temperature",
        display_name="Maximum 2m Air Temperature",
        supported_features=[""],
        description="Maximum air temperature at 2m height (daily maximum)",
        dataset="era5",
    ),
    ChapDataSource(
        name="dewpoint_2m_temperature",
        display_name="Dewpoint 2m Temperature",
        supported_features=[""],
        description="Dewpoint temperature at 2m height (daily average)",
        dataset="era5",
    ),
    ChapDataSource(
        name="total_precipitation",
        display_name="Total Precipitation",
        supported_features=["rainfall"],
        description="Total precipitation (daily sums)",
        dataset="era5",
    ),
    ChapDataSource(
        name="surface_pressure",
        display_name="Surface Pressure",
        supported_features=["surface_pressure"],
        description="Surface pressure (daily average)",
        dataset="era5",
    ),
    ChapDataSource(
        name="mean_sea_level_pressure",
        display_name="Mean Sea Level Pressure",
        supported_features=["mean_sea_level_pressure"],
        description="Mean sea level pressure (daily average)",
        dataset="era5",
    ),
    ChapDataSource(
        name="u_component_of_wind_10m",
        display_name="U Component of Wind 10m",
        supported_features=["u_component_of_wind_10m"],
        description="10m u-component of wind (daily average)",
        dataset="era5",
    ),
    ChapDataSource(
        name="v_component_of_wind_10m",
        display_name="V Component of Wind 10m",
        supported_features=["v_component_of_wind_10m"],
        description="10m v-component of wind (daily average)",
        dataset="era5",
    ),
]


@router.get("/data-sources", response_model=List[ChapDataSource])
async def get_data_sources() -> List[ChapDataSource]:
    return data_sources


@router.post("/create-backtest-with-data/", response_model=ImportSummaryResponse)
async def create_backtest_with_data(
    request: MakeBacktestWithDataRequest,
    dry_run: bool = Query(
        False, description="If True, only run validation and do not create a backtest", alias="dryRun"
    ),
    database_url: str = Depends(get_database_url),
    worker_settings=Depends(get_settings),
):
    feature_names, provided_data_processed = _read_dataset(request)
    provided_data_processed, rejections = _validate_full_dataset(feature_names, provided_data_processed)
    polygon_rejected = provided_data_processed.set_polygons(FeatureCollectionModel.model_validate(request.geojson))
    rejections.extend(
        ValidationError(reason="Missing polygon in geojson", orgUnit=location, feature_name="polygon", time_periods=[])
        for location in polygon_rejected
    )
    imported_count = len(provided_data_processed.locations())
    if dry_run:
        return ImportSummaryResponse(id=None, imported_count=imported_count, rejected=rejections)

    if imported_count == 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Missing values. No data was imported.",
                "rejected": [r.model_dump() for r in rejections],
            },
        )

    logger.info(
        f"Creating backtest with data: {request.name}, model_id: {request.model_id} on {len(provided_data_processed.locations())} locations"
    )
    if request.data_to_be_fetched:
        raise HTTPException(
            status_code=400, detail="data_to_be_fetched is not supported when providing data for backtest"
        )

    bt_params = BackTestParams(**request.model_dump()).model_dump()
    dataset_create_info = DataSetCreateInfo(**request.model_dump()).model_dump()
    dataset_create_info["type"] = "evaluation"
    job = worker.queue_db(
        wf.run_backtest_from_dataset,
        feature_names=feature_names,
        provided_data_model_dump=provided_data_processed.model_dump(),
        dataset_info=dataset_create_info,
        backtest_name=request.name,
        model_id=request.model_id,
        backtest_params=bt_params,
        database_url=database_url,
        worker_config=worker_settings,
        **{JOB_TYPE_KW: "create_backtest_from_data", JOB_NAME_KW: request.name},
    )
    job_id = job.id
    return ImportSummaryResponse(id=job_id, imported_count=imported_count, rejected=rejections)
