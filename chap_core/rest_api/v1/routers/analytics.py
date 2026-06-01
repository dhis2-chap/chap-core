import logging
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy.orm import selectinload
from sqlmodel import Field, Session, select

import chap_core.rest_api.db_worker_functions as wf
from chap_core.api_types import (
    BacktestParams,
    DataElement,
    DataList,
    EvaluationEntry,
    FeatureCollectionModel,
    PredictionEntry,
)
from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.assessment.thresholds import get_threshold_strategy, list_threshold_strategies
from chap_core.database.base_tables import DBModel
from chap_core.database.dataset_manager import DataSetManager
from chap_core.database.dataset_tables import DataSet as DataSetTable
from chap_core.database.dataset_tables import DataSetCreateInfo
from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB
from chap_core.database.tables import Backtest, BacktestForecast, Prediction
from chap_core.datatypes import create_tsdataclass
from chap_core.spatio_temporal_data.converters import observations_to_dataframe, observations_to_dataset
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

from ...celery_tasks import JOB_NAME_KW, JOB_TYPE_KW, CeleryPool, JobType
from ...data_models import (
    BacktestCreate,
    BacktestDomain,
    BacktestRead,
    ChapDataSource,
    DatasetMakeRequest,
    ImportSummaryResponse,
    JobResponse,
    MakeBacktestRequest,
    MakeBacktestWithDataRequest,
    MakePredictionRequest,
    PredictionParams,
    ValidationError,
)
from .dependencies import get_database_url, get_session, get_settings

router = APIRouter(prefix="/analytics")

logger = logging.getLogger(__name__)
worker: CeleryPool[Any] = CeleryPool()


@router.post(
    "/make-dataset",
    response_model=ImportSummaryResponse,
    tags=["Datasets"],
    summary="Import observations as a reusable dataset",
)
def make_dataset(
    request: DatasetMakeRequest, database_url: str = Depends(get_database_url), worker_settings=Depends(get_settings)
):
    """Persist observations (with polygons) as a named dataset you can reuse across backtests and predictions.

    Use this to turn a one-off batch of DHIS2 / file-based observations into something
    stored, so a single dataset can back multiple evaluations. Import happens in the
    background — the response gives you a job id plus a per-location rejection summary
    (validation runs synchronously, the harmonise-and-load step async). Poll
    ``/v1/jobs/{id}`` to know when the dataset is queryable.
    """
    feature_names, provided_data = _read_dataset(request)
    provided_data, rejections = validate_full_dataset(feature_names, provided_data)
    # provided_field_names = {entry.element_id: entry.element_name for entry in request.provided_data}
    polygon_rejected = provided_data.set_polygons(FeatureCollectionModel.model_validate(request.geojson))
    rejections.extend(
        ValidationError(reason="Missing polygon in geojson", orgUnit=location, feature_name="polygon", time_periods=[])
        for location in polygon_rejected
    )
    imported_count = len(list(provided_data.locations()))
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
        **{JOB_TYPE_KW: JobType.DATASET, JOB_NAME_KW: request.name},
    )

    return ImportSummaryResponse(id=job.id, imported_count=imported_count, rejected=rejections)


def _read_dataset(request):
    if not request.provided_data:
        raise HTTPException(status_code=400, detail="No observation data provided.")
    feature_names = list({entry.feature_name for entry in request.provided_data})
    dataclass = create_tsdataclass(feature_names)
    provided_data = observations_to_dataset(dataclass, request.provided_data, fill_missing=True)
    if "population" in feature_names:
        provided_data = provided_data.interpolate(["population"])
    return feature_names, provided_data


def _find_locations_with_complete_covariates(
    dataset: DataSet, feature_names: list[str], target_name: str = "disease_cases"
) -> tuple[set[str], list[ValidationError]]:
    """Identify locations that have no NaN values in covariate features."""
    locations_to_keep = set()
    rejected_list = []
    for location, data in dataset.items():
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
            locations_to_keep.add(location)
    return locations_to_keep, rejected_list


def validate_full_dataset(
    feature_names, provided_data, target_name="disease_cases"
) -> tuple[DataSet, list[ValidationError]]:
    n_locations = len(provided_data.locations())
    locations_to_keep, rejected_list = _find_locations_with_complete_covariates(
        provided_data, feature_names, target_name
    )
    if not locations_to_keep:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "All regions rejected due to missing values",
                "imported_count": 0,
                "rejected": [r.model_dump(by_alias=True) for r in rejected_list],
            },
        )

    new_dataset = _filter_dataset_by_locations(provided_data, locations_to_keep)
    new_locations = list(new_dataset.locations())
    logger.info(
        f"Remaining dataset after validation: {len(new_locations)} (from {n_locations}) locations: {new_locations} "
    )
    assert len(new_locations), new_locations
    return new_dataset, rejected_list


def _find_locations_with_target_data(
    dataset: DataSet,
    target_name: str = "disease_cases",
) -> tuple[set[str], list[ValidationError]]:
    """Identify locations that have at least some non-NaN target data.

    Returns the set of locations to keep and validation errors for rejected ones.
    """
    locations_to_keep = set()
    rejected = []
    for location, data in dataset.items():
        target = getattr(data, target_name)
        if np.any(np.logical_not(np.isnan(target))):
            locations_to_keep.add(location)
        else:
            rejected.append(
                ValidationError(
                    reason="No disease data in the first training split",
                    orgUnit=location,
                    feature_name=target_name,
                    time_periods=[],
                )
            )
    return locations_to_keep, rejected


def _filter_dataset_by_locations(
    dataset: DataSet,
    locations_to_keep: set[str],
) -> DataSet:
    """Return a new dataset containing only the specified locations."""
    new_data = {location: data for location, data in dataset.items() if location in locations_to_keep}
    return dataset.__class__(new_data, polygons=dataset.polygons, metadata=dataset.metadata)


@router.get(
    "/compatible-backtests/{backtestId}",
    response_model=list[BacktestRead],
    tags=["Backtests"],
    summary="Find backtests that can be compared with this one",
)
def get_compatible_backtests(
    backtest_id: Annotated[int, Path(alias="backtestId")], session: Session = Depends(get_session)
):
    """Find every other backtest that shares at least one region and one split period with this one — i.e. backtests it makes sense to overlay or diff against in a plot.

    Use this to power a "compare to..." picker in the UI without offering choices that
    would produce empty intersections.
    """
    logger.info(f"Checking compatible backtests for {backtest_id}")
    backtest = session.get(Backtest, backtest_id)
    if backtest is None:
        raise HTTPException(status_code=404, detail="Backtest not found")
    org_units = set(backtest.org_units)
    split_periods = set(backtest.split_periods)
    res = session.exec(
        select(Backtest.id, Backtest.org_units, Backtest.split_periods).where(Backtest.id != backtest_id)
    ).all()
    ids = [bt_id for bt_id, o, s in res if set(o) & org_units and set(s) & split_periods]
    backtests = session.exec(
        select(Backtest)
        .where(Backtest.id.in_(ids))  # type: ignore[union-attr, attr-defined]
        .options(
            selectinload(Backtest.dataset).defer(DataSetTable.geojson),  # type: ignore[arg-type]
            selectinload(Backtest.configured_model).selectinload(ConfiguredModelDB.model_template),  # type: ignore[arg-type]
        )
    ).all()
    return backtests


@router.get(
    "/backtest-overlap/{backtestId1}/{backtestId2}",
    response_model=BacktestDomain,
    tags=["Backtests"],
    summary="Inspect the shared regions and periods of two backtests",
)
def get_backtest_overlap(
    backtest_id1: Annotated[int, Path(alias="backtestId1")],
    backtest_id2: Annotated[int, Path(alias="backtestId2")],
    session: Session = Depends(get_session),
):
    """Get the regions and split periods two backtests have in common — the slice on which side-by-side comparison is even meaningful.

    Use this before building a side-by-side plot to know which axes are valid for both
    backtests. 404 if either id is unknown.
    """
    backtest1 = session.get(Backtest, backtest_id1)
    backtest2 = session.get(Backtest, backtest_id2)
    if backtest1 is None:
        raise HTTPException(status_code=404, detail=f"Backtest {backtest_id1} not found")
    if backtest2 is None:
        raise HTTPException(status_code=404, detail=f"Backtest {backtest_id2} not found")
    org_units1 = list(set(backtest1.org_units) & set(backtest2.org_units))
    split_periods1 = list(set(backtest1.split_periods) & set(backtest2.split_periods))
    return BacktestDomain(org_units=org_units1, split_periods=split_periods1)


@router.get(
    "/prediction-entry",
    response_model=list[PredictionEntry],
    tags=["Predictions"],
    summary="Read forecast quantiles for a prediction (query parameter id)",
)
async def get_prediction_entry(
    prediction_id: Annotated[int, Query(alias="predictionId")],
    quantiles: list[float] = Query(...),
    session: Session = Depends(get_session),
):
    """Pull the chosen quantiles (median, 10th and 90th percentile, ...) out of a prediction so they can be charted, exported, or fed back into DHIS2.

    Returns one entry per (period, region, quantile) tuple. Same response as
    ``GET /v1/analytics/prediction-entry/{predictionId}`` — only the parameter style
    differs; that variant takes the id in the path. 404 if the prediction id is unknown.
    """
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return [
        PredictionEntry(
            period=forecast.period,
            orgUnit=forecast.org_unit,
            quantile=q,
            value=float(np.quantile(forecast.values, q)),
        )
        for forecast in prediction.forecasts
        for q in quantiles
    ]


@router.get(
    "/evaluation-entry",
    response_model=list[EvaluationEntry],
    tags=["Backtests"],
    summary="Read forecast quantiles from a backtest",
)
async def get_evaluation_entries(
    backtest_id: Annotated[int, Query(alias="backtestId")],
    quantiles: list[float] = Query(...),
    split_period: str = Query(None, alias="splitPeriod"),
    org_units: list[str] = Query(None, alias="orgUnits"),
    session: Session = Depends(get_session),
):
    """Pull the chosen quantiles from a backtest's forecasts so they can be charted, exported, or compared against actuals.

    Optionally narrow the slice with ``splitPeriod`` (a single training horizon) or
    ``orgUnits`` (specific regions). Passing ``orgUnits=["adm0"]`` collapses the result
    into one row per period — the sum over every region — which is what you want for
    national-level overlays. 404 if the backtest id is unknown.
    """
    return_summed = False
    if org_units is not None and len(org_units) == 1 and org_units[0] == "adm0":
        # returning sum of forecasts for all regions
        return_summed = True

    logger.info(
        f"Backtest ID: {backtest_id}, Quantiles: {quantiles}, Split Period: {split_period}, Org Units: {org_units} "
    )
    if backtest_id is not None:
        backtest = session.get(Backtest, backtest_id)
    if backtest is None:
        raise HTTPException(status_code=404, detail="Backtest not found")
    org_units_set: set[str] | None = None
    if org_units is not None:
        org_units_set = set(org_units)
        logger.info("Filtering evaluation entries to org_units: %s", org_units_set)

    cls = BacktestForecast
    expr = select(cls).where(cls.backtest_id == backtest_id)
    if split_period:
        expr = expr.where(cls.last_seen_period == split_period)
    if org_units_set and not return_summed:
        expr = expr.where(cls.org_unit.in_(org_units_set))  # type: ignore[attr-defined]
    forecasts_result = session.exec(expr)

    logger.info(forecasts_result)

    forecasts_list: list[BacktestForecast]
    if return_summed:
        # sum forecasts over all regions
        summed_forecasts: dict[tuple[str, str], Any] = {}
        for forecast in forecasts_result:
            key = (forecast.period, forecast.last_seen_period)
            if key not in summed_forecasts:
                summed_forecasts[key] = np.array([0.0] * len(forecast.values))
            summed_forecasts[key] += np.array(forecast.values)

        forecasts_list = [
            BacktestForecast(
                period=key[0],
                org_unit="adm0",
                last_seen_period=key[1],
                values=values.tolist(),
            )
            for key, values in summed_forecasts.items()
        ]
    else:
        forecasts_list = list(forecasts_result)

    return [
        EvaluationEntry(
            period=forecast.period,
            orgUnit=forecast.org_unit,
            quantile=q,
            splitPeriod=forecast.last_seen_period,
            value=float(np.quantile(forecast.values, q)),
        )
        for forecast in forecasts_list
        for q in quantiles
    ]


@router.post(
    "/create-backtest",
    response_model=JobResponse,
    tags=["Backtests"],
    summary="Run a backtest against a stored dataset",
)
async def create_backtest(
    request: MakeBacktestRequest,
    database_url: str = Depends(get_database_url),
    session: Session = Depends(get_session),
):
    """Train and evaluate a configured model on a dataset that's already been imported, producing a backtest you can score, plot, or promote into a prediction setup.

    Runs asynchronously; you get a job id and poll ``/v1/jobs/{id}`` (or
    ``/v1/jobs/{id}/evaluation_result`` once finished) to find the resulting backtest.
    404 if the referenced dataset does not exist.
    """
    if session.get(DataSetTable, request.dataset_id) is None:
        raise HTTPException(status_code=404, detail=f"Dataset {request.dataset_id} not found")
    job = worker.queue_db(
        wf.run_backtest,
        BacktestCreate(name=request.name, dataset_id=request.dataset_id, model_id=request.model_id),
        request.n_periods,
        request.n_splits,
        request.stride,
        database_url=database_url,
        **{JOB_TYPE_KW: JobType.EVALUATION_LEGACY, JOB_NAME_KW: request.name},
    )

    return JobResponse(id=job.id)


@router.post(
    "/make-prediction",
    response_model=JobResponse,
    tags=["Predictions"],
    summary="Run a one-off forecast from inline data",
)
async def make_prediction(
    request: MakePredictionRequest, database_url=Depends(get_database_url), worker_settings=Depends(get_settings)
):
    """Run a forecast against observations supplied directly in the request body — no stored dataset needed.

    Use this for ad-hoc work when DHIS2 (or another source) already has the data and you
    want it run through a configured model once. Forecasting happens in the background;
    the response carries a job id you poll via ``/v1/jobs/{id}`` for the result. For a
    recurring or scheduled version of the same workflow, use a prediction setup. The
    legacy ``data_to_be_fetched`` field is rejected — supply the observations directly.
    """
    request.type = "prediction"
    feature_names = list({entry.feature_name for entry in request.provided_data})
    dataclass = create_tsdataclass(feature_names)
    provided_data = observations_to_dataset(dataclass, request.provided_data, fill_missing=True)
    if "population" in feature_names:
        provided_data = provided_data.interpolate(["population"])
    provided_data, _rejections = validate_full_dataset(feature_names, provided_data)
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
        **{JOB_TYPE_KW: JobType.PREDICTION, JOB_NAME_KW: request.name},
    )
    return JobResponse(id=job.id)


@router.get(
    "/prediction-entry/{predictionId}",
    response_model=list[PredictionEntry],
    tags=["Predictions"],
    summary="Read forecast quantiles for a prediction (path parameter id)",
)
def get_prediction_entries(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    quantiles: list[float] = Query(...),
    session: Session = Depends(get_session),
):
    """Path-parameter variant of ``GET /v1/analytics/prediction-entry`` for callers that prefer the id in the URL.

    Same forecast-quantile response — pick whichever URL shape your client tooling
    handles more naturally. 404 if the prediction id is unknown.
    """
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return [
        PredictionEntry(
            period=forecast.period, orgUnit=forecast.org_unit, quantile=q, value=float(np.quantile(forecast.values, q))
        )
        for forecast in prediction.forecasts
        for q in quantiles
    ]


@router.get(
    "/actual-cases/{backtestId}",
    response_model=DataList,
    tags=["Backtests"],
    name="get_actual_cases_alias",
    summary="Read the observed disease cases a backtest was scored against",
)
@router.get(
    "/actualCases/{backtestId}",
    response_model=DataList,
    tags=["Backtests"],
    deprecated=True,
    summary="Deprecated camelCase alias of /actual-cases/{backtestId}",
    description=(
        "Deprecated camelCase alias of ``GET /v1/analytics/actual-cases/{backtestId}``. "
        "Behaviour is identical; new integrations should call the kebab-case path, which "
        "matches the rest of the API's URL style."
    ),
)
async def get_actual_cases(
    backtest_id: Annotated[int, Path(alias="backtestId")],
    org_units: list[str] = Query(None, alias="orgUnits"),
    is_dataset_id: bool = Query(False, alias="isDatasetId"),
    session: Session = Depends(get_session),
):
    """Pull the actual ``disease_cases`` series from the dataset that backs a backtest, so a plot can show forecast vs. reality on the same axes.

    Filter to specific regions with ``orgUnits``, or pass ``orgUnits=["adm0"]`` to get
    one summed series across every region (useful for national-level views). Set
    ``isDatasetId=true`` to skip the backtest lookup and read directly from a dataset
    id. 404 if the backtest is unknown (and ``isDatasetId`` is false).
    """
    return_summed = False
    if org_units is not None and len(org_units) == 1 and org_units[0] == "adm0":
        # returning sum of forecasts for all regions
        return_summed = True
    if not is_dataset_id:
        backtest = session.get(Backtest, backtest_id)
        logger.info(f"Backtest: {backtest}")
        if backtest is None:
            raise HTTPException(status_code=404, detail="Backtest not found")
        dataset_id = backtest.dataset_id
    else:
        dataset_id = backtest_id
    observations = DataSetManager(session).observations(
        dataset_id,
        org_units=None if return_summed else org_units,
        feature_names=["disease_cases"],
    )
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


@router.get(
    "/data-sources",
    response_model=list[ChapDataSource],
    tags=["Datasets"],
    summary="Discover which external data sources can feed a dataset",
)
async def get_data_sources() -> list[ChapDataSource]:
    """List the external covariate sources CHAP knows about (ERA5 temperature, precipitation, ...) and the dataset features each one maps to.

    Use this to power a data-source picker when authoring a dataset or a prediction
    setup, so users can pick covariates by what they cover rather than by their internal
    identifier.
    """
    return data_sources


@router.post(
    "/create-backtest-with-data/",
    response_model=ImportSummaryResponse,
    tags=["Backtests"],
    summary="Run a backtest from inline data",
)
async def create_backtest_with_data(
    request: MakeBacktestWithDataRequest,
    dry_run: bool = Query(
        False, description="If True, only run validation and do not create a backtest", alias="dryRun"
    ),
    database_url: str = Depends(get_database_url),
    worker_settings=Depends(get_settings),
):
    """Train and evaluate a model on observations supplied directly in the request body, without first creating a reusable dataset.

    Convenient for quick experiments where the data is not worth persisting. Pass
    ``dryRun=true`` to run validation only (cheap, synchronous) and inspect which
    regions would be rejected — useful as a preflight before committing to an import. A
    real run returns a job id; poll ``/v1/jobs/{id}`` for status. The response also
    surfaces any per-location rejections that came out of validation.
    """
    try:
        feature_names, provided_data_processed = _read_dataset(request)
        provided_data_processed, rejections = validate_full_dataset(feature_names, provided_data_processed)
    except HTTPException as exc:
        if not dry_run or exc.status_code != 400:
            raise
        # Rejections, when present, were serialised by `validate_full_dataset` into
        # `exc.detail["rejected"]`. The empty-`provided_data` case in `_read_dataset`
        # raises with a plain-string detail and has no rejections to recover. If
        # either helper changes its detail shape, this branch must be updated.
        rejected: list = exc.detail.get("rejected", []) if isinstance(exc.detail, dict) else []
        return ImportSummaryResponse.model_validate({"id": None, "imported_count": 0, "rejected": rejected})
    backtest_params = BacktestParams(**request.model_dump())
    train_set, _ = train_test_generator(
        provided_data_processed, backtest_params.n_periods, backtest_params.n_splits, stride=backtest_params.stride
    )
    locations_to_keep, target_rejections = _find_locations_with_target_data(train_set)
    provided_data_processed = _filter_dataset_by_locations(provided_data_processed, locations_to_keep)
    rejections.extend(target_rejections)
    polygon_rejected = provided_data_processed.set_polygons(FeatureCollectionModel.model_validate(request.geojson))
    rejections.extend(
        ValidationError(reason="Missing polygon in geojson", orgUnit=location, feature_name="polygon", time_periods=[])
        for location in polygon_rejected
    )
    imported_count = len(list(provided_data_processed.locations()))
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
        f"Creating backtest with data: {request.name}, model_id: {request.model_id} on {len(list(provided_data_processed.locations()))} locations"
    )
    if request.data_to_be_fetched:
        raise HTTPException(
            status_code=400, detail="data_to_be_fetched is not supported when providing data for backtest"
        )

    bt_params = BacktestParams(**request.model_dump()).model_dump()
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
        **{JOB_TYPE_KW: JobType.EVALUATION, JOB_NAME_KW: request.name},
    )
    job_id = job.id
    return ImportSummaryResponse(id=job_id, imported_count=imported_count, rejected=rejections)


class ThresholdRequest(DBModel):
    """Request body for computing thresholds (endemic channel) for a dataset."""

    dataset_id: int = Field(description="Primary key of the dataset to compute thresholds from.")
    period_ids: list[str] = Field(description='Periods to produce a threshold for, e.g. `["2024-01", "2024-02"]`.')
    strategy: str = Field(description="Registered threshold strategy id (see GET /thresholds/strategies).")
    params: dict = Field(default={}, description="Optional strategy-specific parameters.")


class ThresholdEntry(DBModel):
    """One computed threshold for a single (period, org unit)."""

    period: str = Field(description="Period the threshold applies to.")
    orgUnit: str = Field(description="Org unit the threshold applies to.")
    value: float | None = Field(description="Computed threshold value, or `None` if it could not be computed.")


class ThresholdStrategyInfo(DBModel):
    """Catalogue entry for one registered threshold strategy."""

    id: str = Field(description="Canonical strategy identifier used in request bodies.")
    display_name: str = Field(description="Human-friendly strategy name shown in pickers.")
    description: str = Field(default="", description="Short paragraph explaining what the strategy computes.")


@router.get(
    "/thresholds/strategies",
    response_model=list[ThresholdStrategyInfo],
    tags=["Datasets"],
    summary="Discover which threshold strategies are available",
)
def list_threshold_strategy_types():
    """List the registered threshold strategies (seasonal mean + k*std, ...), with a name and description for each.

    Use this to populate a strategy picker before requesting a specific threshold via
    `POST /v1/analytics/thresholds`.
    """
    return [
        ThresholdStrategyInfo(id=s["id"], display_name=s["name"], description=s["description"])
        for s in list_threshold_strategies()
    ]


@router.post(
    "/thresholds",
    response_model=list[ThresholdEntry],
    tags=["Datasets"],
    summary="Compute thresholds (endemic channel) for a dataset",
)
def compute_thresholds(request: ThresholdRequest, session: Session = Depends(get_session)):
    """Compute one outbreak threshold per (period, org unit) from a dataset's historical disease_cases, using the chosen strategy.

    404 if the strategy id is not registered or the dataset has no `disease_cases`
    observations.
    """
    strategy_cls = get_threshold_strategy(request.strategy)
    if strategy_cls is None:
        available = ", ".join(s["id"] for s in list_threshold_strategies())
        raise HTTPException(
            status_code=404, detail=f"Unknown threshold strategy: {request.strategy}. Available: {available}"
        )

    observations = DataSetManager(session).observations(request.dataset_id, feature_names=["disease_cases"])
    if not observations:
        raise HTTPException(
            status_code=404, detail=f"No disease_cases observations found for dataset {request.dataset_id}"
        )

    df = observations_to_dataframe(observations).rename(columns={"value": "disease_cases"})[
        ["location", "time_period", "disease_cases"]
    ]
    result = strategy_cls().compute(df, request.period_ids, request.params)
    return [
        ThresholdEntry(
            period=str(row["period_id"]),
            orgUnit=str(row["org_unit"]),
            value=None if (row["threshold"] is None or np.isnan(row["threshold"])) else float(row["threshold"]),
        )
        for row in result.to_dict("records")
    ]
