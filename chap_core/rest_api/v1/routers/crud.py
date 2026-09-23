"""
This module contains rest api endpoints for CRUDish operations on the database
Create/Post endpoints will either return the database id of the created object or a job id
that can later be used to retrieve the database id of the created object.

List endpoints will return a list of objects in the database without full data

Get endpoints will return a single object with full data

We try to make the returned objects look as much as possible like the objects in the database
This is achieved by subclassing common basemodels in the read objects and database table objects

Returned objects come out camelCase while internal objects stay snake_case because DBModel sets
alias_generator=to_camel and FastAPI's response_model_by_alias defaults to True.

"""

import json
import logging
from typing import Annotated, Any, Final

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Path, Query, UploadFile
from sqlalchemy import func
from sqlalchemy.orm import selectinload
from sqlmodel import Session, col, select
from starlette.responses import StreamingResponse

import chap_core.rest_api.db_worker_functions as wf
from chap_core.api_types import BacktestParams, FeatureCollectionModel
from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.metrics import compute_all_detailed_metrics
from chap_core.assessment.weather_providers import resolve_weather_provider
from chap_core.data import DataSet as InMemoryDataSet
from chap_core.database.database import SessionWrapper
from chap_core.database.dataset_manager import DataSetManager
from chap_core.database.dataset_tables import (
    DataSet,
    DataSetCreateInfo,
    DataSetInfo,
    DataSetWithObservations,
)
from chap_core.database.model_spec_tables import ModelSpecRead
from chap_core.database.model_templates_and_config_tables import (
    ConfiguredModelDB,
    ModelConfiguration,
    ModelTemplateDB,
    chapkit_revision_conflict,
)
from chap_core.database.tables import (
    Backtest,
    BacktestSpecification,
    Prediction,
    PredictionInfo,
    PredictionSetupRead,
    PredictionSetupReadWithPredictions,
)
from chap_core.datatypes import FullData, HealthPopulationData, create_tsdataclass
from chap_core.exceptions import ModelTemplateRevisionConflict
from chap_core.geometry import Polygons
from chap_core.rest_api.celery_tasks import (
    JOB_NAME_KW,
    JOB_TYPE_KW,
    PREDICTION_SETUP_ID_JOB_META_KEY,
    CeleryPool,
    JobType,
)
from chap_core.rest_api.celery_tasks import r as redis
from chap_core.rest_api.experimental import api_experimental
from chap_core.rest_api.services.orchestrator import Orchestrator, ServiceNotFoundError
from chap_core.rest_api.services.schemas import MLServiceInfo
from chap_core.rest_api.v2.dependencies import get_orchestrator
from chap_core.services import prediction_setup_service
from chap_core.spatio_temporal_data.converters import observations_to_dataset

from ...data_models import (
    BacktestRead,
    BacktestSpecificationFilter,
    BacktestSpecificationRead,
    BacktestSpecificationSummary,
    BacktestUpdate,
    ConfiguredModelInfoRead,
    DataBaseResponse,
    DatasetCreate,
    JobResponse,
    ModelConfigurationCreate,
    ModelTemplateCreate,
    ModelTemplateFromService,
    ModelTemplateRead,
    PredictionParams,
    PredictionSetupCreate,
    PredictionSetupUpdate,
    RunPredictionSetupRequest,
)
from .analytics import validate_full_dataset
from .dependencies import get_database_url, get_session, get_settings

logger = logging.getLogger(__name__)


LIVE: Final = "live"
REVISION_MISMATCH: Final = "revision_mismatch"


def _registered_chapkit_revision_conflict(
    session: Session, info: MLServiceInfo
) -> ModelTemplateRevisionConflict | None:
    """The conflict between a registered service and the template stored under its version, if any.

    Computed on every read instead of persisted, because the service can change after a
    sync and a redeploy of the right image should clear the state without cleanup.
    """
    template = session.exec(
        select(ModelTemplateDB).where(ModelTemplateDB.name == info.id, ModelTemplateDB.version == info.version)
    ).first()
    if template is not None:
        return chapkit_revision_conflict(template, info.git_revision)
    if info.git_revision is None:
        # Not stored: a row with no digest could never run, and the label would be burnt
        # for the build that does report a revision.
        return ModelTemplateRevisionConflict(
            info.id,
            info.version,
            None,
            None,
            "build the image with the GIT_REVISION build arg and register it again. The version "
            "label is not stored yet, so it can be kept.",
        )
    return None


def _sync_live_chapkit_services(session: Session, orchestrator=None) -> dict[str, ModelTemplateRevisionConflict | None]:
    """Check the live chapkit services in the v2 registry against the stored templates.

    Registration is a liveness and URL signal only. It creates no templates or
    configured models: a service becomes a model in CHAP when it is installed with
    ``chap-admin install`` or seeded at startup. The one thing taken from the live
    service is a template's user option schema, which the marketplace registry does
    not carry, the first time the service is reachable.

    Callers that already hold an orchestrator should pass it in. Building a
    fresh one here reaches around FastAPI's dependency overrides and opens a
    second redis connection, which costs a full connect timeout whenever
    redis is unreachable.

    Returns the revision conflict per registered service id, or None when the
    service runs the stored source revision. Silently returns nothing if Redis is unavailable.
    """
    try:
        if orchestrator is None:
            from chap_core.rest_api.v2.dependencies import get_orchestrator

            orchestrator = get_orchestrator()
        service_list = orchestrator.get_all()
    except Exception:
        logger.debug("Could not reach service registry, skipping chapkit sync")
        return {}

    conflicts: dict[str, ModelTemplateRevisionConflict | None] = {}
    for service in service_list.services:
        conflict = _registered_chapkit_revision_conflict(session, service.info)
        conflicts[service.info.id] = conflict
        if conflict is not None:
            logger.warning(str(conflict))
            continue
        try:
            _fill_user_options_from_service(session, service)
        except Exception:
            logger.warning("Could not fetch config schema from %s, will retry next sync", service.url, exc_info=True)
    return conflicts


def _fill_user_options_from_service(session: Session, service) -> None:
    """Complete a stored template's user option schema from its live service, once."""
    template = session.exec(
        select(ModelTemplateDB).where(
            ModelTemplateDB.name == service.info.id, ModelTemplateDB.version == service.info.version
        )
    ).first()
    if template is None or template.user_options:
        return
    user_options = _fetch_user_options(service.url)
    if user_options:
        template.user_options = user_options
        session.add(template)
        session.commit()


def _fetch_user_options(service_url: str) -> dict:
    from chap_core.models.chapkit_rest_api_wrapper import CHAPKitRestAPIWrapper
    from chap_core.models.external_chapkit_model import _parse_user_options_from_config_schema

    client = CHAPKitRestAPIWrapper(service_url, timeout=5)
    try:
        return _parse_user_options_from_config_schema(client.get_config_schema())
    finally:
        client.close()


def add_model_template_from_registered_service(session: Session, service) -> int:
    """Store the template a registered chapkit service describes, from its live info and config schema.

    This is the path for custom images without a marketplace entry. The service must
    report a git revision, since a template without a digest could never run, and a
    stored version must be the revision the service reports.
    """
    from chap_core.models.external_chapkit_model import ml_service_info_to_model_template_config

    conflict = _registered_chapkit_revision_conflict(session, service.info)
    if conflict is not None:
        raise conflict
    config = ml_service_info_to_model_template_config(service.info, service.url, _fetch_user_options(service.url))
    wrapper = SessionWrapper(session=session)
    template_id = wrapper.add_model_template_from_yaml_config(config, source_digest=service.info.git_revision)
    template = wrapper.get_model_template(template_id)
    template.uses_chapkit = True
    session.commit()
    return template_id


router = APIRouter(prefix="/crud")

worker: CeleryPool[Any] = CeleryPool()


###########
# backtests


@router.get(
    "/backtests",
    response_model=list[BacktestRead],
    tags=["Backtests"],
    summary="Browse stored evaluation runs",
)  # This should be called list
async def get_backtests(
    specification_id: Annotated[int | None, Query(alias="specificationId")] = None,
    dataset_id: Annotated[int | None, Query(alias="datasetId")] = None,
    session: Session = Depends(get_session),
):
    """List stored backtests so you can pick one to view, compare against another, plot metrics from, or promote into a saved prediction setup.

    Each entry carries enough metadata to identify it at a glance (dataset, model,
    periods, regions) but not the raw forecasts — fetch those via
    ``/backtests/{id}/full`` only when you actually need them. Filter by
    ``specificationId`` to get the backtests that are comparable with each other, or by
    ``datasetId`` for everything run against one dataset.
    """
    query = select(Backtest).options(*_backtest_read_loads())
    if specification_id is not None:
        query = query.where(Backtest.specification_id == specification_id)
    if dataset_id is not None:
        query = query.where(Backtest.dataset_id == dataset_id)
    return session.exec(query).all()


def _backtest_read_loads():
    """Eager loads for everything `BacktestRead` reads off a `Backtest` row."""
    return (
        selectinload(Backtest.specification),  # type: ignore[arg-type]
        selectinload(Backtest.dataset).defer(DataSet.geojson),  # type: ignore[arg-type]
        selectinload(Backtest.configured_model).selectinload(ConfiguredModelDB.model_template),  # type: ignore[arg-type]
        selectinload(Backtest.prediction_setup),  # type: ignore[arg-type]
    )


@router.get(
    "/backtest-specifications",
    response_model=list[BacktestSpecificationSummary],
    tags=["Backtests"],
    summary="List the evaluation setups backtests have run under",
)
def get_backtest_specifications(
    filters: Annotated[BacktestSpecificationFilter, Query()],
    session: Session = Depends(get_session),
):
    """List backtest specifications: a dataset plus the parameters that make backtests under it comparable.

    A specification with several backtests under it is a benchmark. Filter by
    ``datasetId`` and any of the ``BacktestParams`` fields; the full tuple identifies at
    most one specification, which is how an external system finds a benchmark again
    without storing the specification id. Rows carry counts only; fetch
    ``/backtest-specifications/{id}`` for the backtests themselves.
    """
    backtest_counts = (
        select(Backtest.specification_id, func.count(col(Backtest.id)).label("backtest_count"))
        .group_by(col(Backtest.specification_id))
        .subquery()
    )
    query = (
        select(BacktestSpecification, func.coalesce(backtest_counts.c.backtest_count, 0))
        .outerjoin(backtest_counts, backtest_counts.c.specification_id == BacktestSpecification.id)
        .options(selectinload(BacktestSpecification.dataset).defer(DataSet.geojson))  # type: ignore[arg-type]
        .order_by(col(BacktestSpecification.id))
    )
    for name, value in filters.model_dump(exclude_none=True).items():
        query = query.where(getattr(BacktestSpecification, name) == value)
    return [
        BacktestSpecificationSummary(
            id=specification.id,
            dataset=specification.dataset,
            org_unit_count=len(specification.org_units),
            backtest_count=backtest_count,
            **_specification_params(specification),
        )
        for specification, backtest_count in session.exec(query).all()
    ]


@router.get(
    "/backtest-specifications/{specificationId}",
    response_model=BacktestSpecificationRead,
    tags=["Backtests"],
    summary="Fetch a specification with every backtest under it",
)
def get_backtest_specification(
    specification_id: Annotated[int, Path(alias="specificationId")], session: Session = Depends(get_session)
):
    """Read one specification together with every backtest that ran under it, newest first, in a single response.

    This is the benchmark leaderboard: each backtest row is the ``BacktestRead`` shape
    with aggregate metrics, the configured model and its template, so a client can rank
    models without a request per backtest. Forecasts and per-org-unit metrics are not
    included. 404 if the id is unknown.
    """
    specification = session.exec(
        select(BacktestSpecification)
        .where(BacktestSpecification.id == specification_id)
        .options(selectinload(BacktestSpecification.dataset).defer(DataSet.geojson))  # type: ignore[arg-type]
    ).first()
    if specification is None:
        raise HTTPException(status_code=404, detail="Backtest specification not found")
    backtests = session.exec(
        select(Backtest)
        .where(Backtest.specification_id == specification_id)
        .order_by(col(Backtest.created).desc().nulls_last(), col(Backtest.id).desc())
        .options(*_backtest_read_loads())
    ).all()
    return BacktestSpecificationRead(
        id=specification_id,
        dataset=specification.dataset,
        org_units=specification.org_units,
        backtests=backtests,
        **_specification_params(specification),
    )


def _specification_params(specification: BacktestSpecification) -> dict[str, Any]:
    return {name: getattr(specification, name) for name in BacktestParams.model_fields}


@router.get(
    "/backtests/{backtestId}/full",
    response_model=Backtest,
    tags=["Backtests"],
    summary="Fetch a backtest with every forecast inline",
)
async def get_backtest(backtest_id: Annotated[int, Path(alias="backtestId")], session: Session = Depends(get_session)):
    """Load the complete backtest payload — every forecast row and the dataset's GeoJSON — in a single response.

    Use this when a client genuinely needs the whole evaluation (e.g. exporting it,
    rebuilding it offline). For listings or UI summaries, the cheaper ``/info`` or
    ``/backtests/{id}`` variants are usually what you want. 404 if the id is unknown.
    """
    backtest = session.get(Backtest, backtest_id)
    if backtest is None:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return backtest


@router.get(
    "/backtests/{backtestId}/info",
    response_model=BacktestRead,
    tags=["Backtests"],
    summary="View one backtest's metadata",
)
@router.get(
    "/backtests/{backtestId}",
    response_model=BacktestRead,
    tags=["Backtests"],
    summary="View one backtest's metadata (alias of /info)",
)
def get_backtest_info(backtest_id: Annotated[int, Path(alias="backtestId")], session: Session = Depends(get_session)):
    """Read a single backtest's identifying information — name, dataset, configured model + template, the periods and regions it covers — without paying for the forecast payload.

    Use this for detail panes, breadcrumb headers, or anywhere a UI needs to render
    "what is this backtest" without scrolling through forecasts. Both
    ``/backtests/{id}`` and ``/backtests/{id}/info`` resolve to this same operation;
    fetch ``/full`` if you also want the forecasts. 404 if the id is unknown.
    """
    backtest = session.exec(
        select(Backtest)
        .where(Backtest.id == backtest_id)
        .options(
            selectinload(Backtest.specification),  # type: ignore[arg-type]
            selectinload(Backtest.dataset).defer(DataSet.geojson),  # type: ignore[arg-type]
            selectinload(Backtest.configured_model).selectinload(ConfiguredModelDB.model_template),  # type: ignore[arg-type]
            selectinload(Backtest.prediction_setup),  # type: ignore[arg-type]
        )
    ).first()
    if backtest is None:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return backtest


@router.get(
    "/metric/csv",
    tags=["Metrics"],
    summary="Export backtest metrics for offline analysis",
)
async def get_metrics_csv(
    backtest_id: Annotated[int, Query(alias="backtestId")],
    session: Session = Depends(get_session),
):
    """Download every scoring metric computed for a backtest as a CSV, broken down by region, time period, and forecast horizon.

    Use this when you want to pull metrics into pandas, Excel, or BI tooling for
    analysis the built-in plots don't cover — for example comparing several backtests
    side by side, or weighting locations differently. The path is scoped to ``/metric/``
    so it can be extended to multi-backtest exports later without breaking callers.
    404 if the backtest is unknown.
    """
    backtest = session.get(Backtest, backtest_id)
    if backtest is None:
        raise HTTPException(status_code=404, detail="Backtest not found")

    evaluation = Evaluation.from_backtest(backtest)
    df = compute_all_detailed_metrics(evaluation)
    df["time_period"] = df["time_period"].astype(str)

    csv_content = df.to_csv(index=False)
    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=backtest_{backtest_id}_metrics.csv"},
    )


@router.delete(
    "/backtests/{backtestId}",
    tags=["Backtests"],
    summary="Remove an evaluation run",
)
async def delete_backtest(
    backtest_id: Annotated[int, Path(alias="backtestId")], session: Session = Depends(get_session)
):
    """Permanently delete a backtest and every forecast attached to it.

    Use this to clean up failed runs or evaluations that should no longer appear in the
    listing. Returns 404 if the id is unknown.
    """
    backtest = session.get(Backtest, backtest_id)
    if backtest is None:
        raise HTTPException(status_code=404, detail="Backtest not found")
    session.delete(backtest)
    session.commit()
    return {"message": "deleted"}


@router.patch(
    "/backtests/{backtestId}",
    response_model=BacktestRead,
    tags=["Backtests"],
    summary="Edit a backtest's editable fields",
)
async def update_backtest(
    backtest_id: Annotated[int, Path(alias="backtestId")],
    backtest_update: BacktestUpdate,
    session: Session = Depends(get_session),
):
    """Rename a backtest or update its mutable metadata without re-running the evaluation.

    Only the fields you send are touched (semantically: ``exclude_unset``), so it is
    safe to PATCH a single attribute. Returns the refreshed metadata. 404 if the id is
    unknown.
    """
    db_backtest = session.get(Backtest, backtest_id)
    if not db_backtest:
        raise HTTPException(status_code=404, detail="Backtest not found")

    update_data = backtest_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_backtest, key, value)

    session.add(db_backtest)
    session.commit()

    # Reload with eager loading to avoid lazy-load issues
    db_backtest = session.exec(
        select(Backtest)
        .where(Backtest.id == backtest_id)
        .options(
            selectinload(Backtest.specification),  # type: ignore[arg-type]
            selectinload(Backtest.dataset).defer(DataSet.geojson),  # type: ignore[arg-type]
            selectinload(Backtest.configured_model).selectinload(ConfiguredModelDB.model_template),  # type: ignore[arg-type]
            selectinload(Backtest.prediction_setup),  # type: ignore[arg-type]
        )
    ).first()
    return db_backtest


@router.delete(
    "/backtests",
    tags=["Backtests"],
    summary="Bulk-remove several evaluation runs",
)
async def delete_backtest_batch(ids: Annotated[str, Query(alias="ids")], session: Session = Depends(get_session)):
    """Permanently delete several backtests in one round-trip — pass their ids as a comma-separated ``ids`` query string.

    Useful for bulk cleanup from a UI's multi-select. Unknown ids are silently skipped;
    the response only reports how many rows actually went away. 400 if ``ids`` is empty
    or contains a non-integer segment.
    """
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
            ) from None

    for backtest_id in backtest_ids_list:
        backtest = session.get(Backtest, backtest_id)
        if backtest is not None:
            session.delete(backtest)
            deleted_count += 1
    session.commit()
    return {"message": f"Deleted {deleted_count} backtests"}


###########
# predictions


@router.get(
    "/predictions",
    response_model=list[PredictionInfo],
    tags=["Predictions"],
    summary="Browse stored forecasts",
)
async def get_predictions(session: Session = Depends(get_session)):
    """List every prediction whose configured model row still exists in the database, so you can pick a forecast to inspect, plot, push back into DHIS2, or delete.

    Predictions whose configured model has been deleted outright are filtered out;
    predictions whose configured model has only been archived (soft-deleted) still
    appear, since the row is still resolvable.
    """
    session_wrapper = SessionWrapper(session=session)
    return [
        prediction for prediction in session_wrapper.list_all(Prediction) if prediction.configured_model is not None
    ]


@router.get(
    "/predictions/{predictionId}",
    response_model=PredictionInfo,
    tags=["Predictions"],
    summary="View one forecast's metadata",
)
async def get_prediction(
    prediction_id: Annotated[int, Path(alias="predictionId")], session: Session = Depends(get_session)
):
    """Read the identifying information for a single prediction — the model and dataset behind it, when it ran, what periods it covers — without pulling the forecast values themselves.

    Use ``GET /v1/analytics/prediction-entry/{id}`` once you actually need quantiles to
    plot. 404 if the id is unknown.
    """
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return prediction


@router.delete(
    "/predictions/{predictionId}",
    tags=["Predictions"],
    summary="Remove a forecast",
)
async def delete_prediction(
    prediction_id: Annotated[int, Path(alias="predictionId")], session: Session = Depends(get_session)
):
    """Permanently delete a prediction and every forecast row it contains. Use this to clean up obsolete or test forecasts from the listing. 404 if the id is unknown."""
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    session.delete(prediction)
    session.commit()
    return {"message": "deleted"}


###########
# datasets


@router.get(
    "/datasets",
    response_model=list[DataSetInfo],
    tags=["Datasets"],
    summary="Browse imported datasets",
)
async def get_datasets(session: Session = Depends(get_session)):
    """List every imported dataset so you can pick one to back a backtest, run a prediction, or inspect its contents — metadata only, no observations inline."""
    datasets = session.exec(select(DataSet)).all()
    return datasets


@router.get(
    "/datasets/{datasetId}",
    response_model=DataSetWithObservations,
    tags=["Datasets"],
    summary="Inspect a dataset's observations",
)
async def get_dataset(dataset_id: Annotated[int, Path(alias="datasetId")], session: Session = Depends(get_session)):
    """Load a dataset together with every observation it contains, so you can audit what was imported or re-export it.

    NaN/inf values are coerced to JSON ``null`` so the response is always parseable.
    Heavier than the listing — for casual browsing, prefer ``/datasets``. 404 if the id
    is unknown.
    """
    # dataset = session.exec(select(DataSet).where(DataSet.id == dataset_id)).first()
    dataset = session.get(DataSet, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    assert len(dataset.observations) > 0
    for obs in dataset.observations:
        obs.value = obs.value if obs.value is None or np.isfinite(obs.value) else None
    return dataset


@router.post(
    "/datasets",
    tags=["Datasets"],
    summary="Import a health-only dataset",
)
async def create_dataset(
    data: DatasetCreate, datababase_url=Depends(get_database_url), worker_settings=Depends(get_settings)
) -> JobResponse:
    """Import a dataset that carries just disease cases and population (no climate covariates inline), with polygons attached.

    Climate or other covariates are layered on later by the modelling pipeline.
    Importing runs in the background; you get a job id and poll ``/v1/jobs/{id}``
    (or ``/v1/jobs/{id}/database_result`` once finished) for the resulting dataset id.
    For a dataset that ships its own covariates, use ``POST /v1/analytics/make-dataset``
    instead.
    """
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


@router.post(
    "/datasets/csvFile",
    tags=["Datasets"],
    summary="Import a dataset from a CSV + geojson upload",
)
async def create_dataset_csv(
    csv_file: UploadFile = File(...),
    geojson_file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> DataBaseResponse:
    """Upload a CSV of observations together with the matching GeoJSON polygons (``NAME_1`` keyed) and persist it as a dataset synchronously.

    Use this when you have the files on disk and do not want to round-trip them
    through DHIS2 or the JSON import endpoints. Inserts inline — no background job —
    and returns the new dataset id immediately.
    """
    import io

    csv_content = await csv_file.read()
    dataset = InMemoryDataSet.from_csv(io.BytesIO(csv_content), dataclass=FullData)
    geo_json_content = await geojson_file.read()
    features = Polygons.from_geojson(json.loads(geo_json_content), id_property="NAME_1").feature_collection()
    dataset_id = DataSetManager(session).save_dataset(
        DataSetCreateInfo(name="csv_file"), dataset, features.model_dump_json()
    )
    return DataBaseResponse(id=dataset_id)


@router.get(
    "/datasets/{datasetId}/df",
    tags=["Datasets"],
    summary="Read a dataset as tabular JSON rows",
)
async def get_dataset_df(dataset_id: Annotated[int, Path(alias="datasetId")], session: Session = Depends(get_session)):
    """Get a dataset shaped as a list of JSON rows (one row per region and time period), in the form pandas, Observable, or any tabular tool can consume directly.

    Use this when a UI needs to render the dataset as a table, or when a notebook
    consumer wants to drop the result into ``pd.DataFrame``. Non-finite values come
    through as JSON ``null``. 404 if the dataset id is unknown.
    """
    if session.get(DataSet, dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    in_memory_dataset = DataSetManager(session).to_dataset(dataset_id)
    df = in_memory_dataset.to_pandas()
    # Convert time_period column to strings for proper serialization
    df["time_period"] = df["time_period"].astype(str)
    records = df.to_dict(orient="records")
    # NaN floats are not JSON-serialisable; surface them as JSON null instead.
    # Done after to_dict because reassigning None into a float column re-casts it to NaN.
    for record in records:
        for key, value in record.items():
            if isinstance(value, float) and not np.isfinite(value):
                record[key] = None
    return records


@router.get(
    "/datasets/{datasetId}/csv",
    tags=["Datasets"],
    summary="Export a dataset as a CSV download",
)
async def get_dataset_csv(dataset_id: Annotated[int, Path(alias="datasetId")], session: Session = Depends(get_session)):
    """Download a dataset as a CSV file — one row per region and time period.

    Use this when a user wants to pull the imported data out for use in Excel, R,
    pandas, or any other offline tool. 404 if the dataset id is unknown.
    """
    if session.get(DataSet, dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    in_memory_dataset = DataSetManager(session).to_dataset(dataset_id)
    df = in_memory_dataset.to_pandas()
    df["time_period"] = df["time_period"].astype(str)

    csv_content = df.to_csv(index=False)
    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=dataset_{dataset_id}.csv"},
    )


@router.delete(
    "/datasets/{datasetId}",
    tags=["Datasets"],
    summary="Remove an imported dataset",
)
async def delete_dataset(dataset_id: Annotated[int, Path(alias="datasetId")], session: Session = Depends(get_session)):
    """Permanently delete a dataset and every observation in it. Use this to clean up obsolete imports — be aware that backtests and predictions that referenced this dataset will lose their data link. 404 if the id is unknown."""
    # dataset = session.exec(select(DataSet).where(DataSet.id == dataset_id)).first()
    if session.get(DataSet, dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    DataSetManager(session).delete_by_id(dataset_id)
    return {"message": "deleted"}


###########
# model templates


@router.get(
    "/model-templates",
    response_model=list[ModelTemplateRead],
    tags=["Models"],
    summary="Browse available model templates",
)
async def list_model_templates(session: Session = Depends(get_session)):
    """List every live model template that can be configured into a runnable model — one per template name; superseded versions keep their rows but are not listed.

    The CHAPKit v2 service registry is checked on the way, so a template's
    ``health_status`` reflects whether the backing CHAPKit service is currently
    registered (``"live"``) and still runs the stored source revision
    (``"revision_mismatch"`` otherwise). Registration alone does not create a
    template; see ``POST /v1/crud/model-templates``.
    """
    conflicts = _sync_live_chapkit_services(session)
    model_templates = session.exec(select(ModelTemplateDB).where(ModelTemplateDB.is_live == True)).all()

    results = []
    for t in model_templates:
        read = ModelTemplateRead.model_validate(t)
        if t.name in conflicts:
            read.health_status = LIVE if conflicts[t.name] is None else REVISION_MISMATCH
        results.append(read)
    return results


@router.post(
    "/model-templates",
    response_model=ModelTemplateRead,
    tags=["Models"],
    summary="Store a model template version",
)
def add_model_template(model_template: ModelTemplateCreate, session: Session = Depends(get_session)):
    """Store a model template version, for example a marketplace model that ``chap-admin install`` registers.

    A version is write-once. Posting a stored name and version again returns the stored
    row unchanged (and shows it again if it was retired), so the call can be repeated.
    Posting it with another source digest is refused with 409: use a new version label.
    """
    wrapper = SessionWrapper(session=session)
    try:
        template_id = wrapper.add_model_template_version(ModelTemplateDB(**model_template.model_dump()))
    except ModelTemplateRevisionConflict as conflict:
        raise HTTPException(status_code=409, detail=str(conflict)) from conflict
    return ModelTemplateRead.model_validate(wrapper.get_model_template(template_id))


@router.post(
    "/model-templates/from-service",
    response_model=ModelTemplateRead,
    tags=["Models"],
    summary="Store a model template from a registered CHAPKit service",
)
def add_model_template_from_service(
    request: ModelTemplateFromService,
    session: Session = Depends(get_session),
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    """Store the template a live CHAPKit service describes, read from its own info and config schema.

    This is how a custom image without a marketplace entry becomes a model in CHAP.
    The service must be registered in the v2 service registry and reachable. 404 if it
    is not registered, 409 if it reports no git revision or another revision than the
    one stored under its version, 502 if it cannot be read.
    """
    try:
        service = orchestrator.get(request.service_id)
    except ServiceNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    try:
        template_id = add_model_template_from_registered_service(session, service)
    except ModelTemplateRevisionConflict as conflict:
        raise HTTPException(status_code=409, detail=str(conflict)) from conflict
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Could not read the service at {service.url}: {error}") from error
    return ModelTemplateRead.model_validate(SessionWrapper(session=session).get_model_template(template_id))


@router.delete(
    "/model-templates/{modelTemplateId}",
    tags=["Models"],
    summary="Retire a model template",
)
def delete_model_template(
    model_template_id: Annotated[int, Path(alias="modelTemplateId")], session: Session = Depends(get_session)
):
    """Hide a model template and its configured models from pickers, keeping the rows so historical backtests still resolve.

    Storing the same name and version again shows the template again. 404 if the id is unknown.
    """
    try:
        SessionWrapper(session=session).archive_model_template(model_template_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {"message": "deleted"}


###########
# configured models


@router.get(
    "/configured-models",
    response_model=list[ModelSpecRead],
    tags=["Models"],
    summary="Browse configured (ready-to-run) models",
)
def list_configured_models(session: Session = Depends(get_session)):
    """List every configured model — a template + user-chosen options bundled into something you can actually run.

    Use this to populate model pickers in backtest / prediction creation flows. Each
    entry carries the configuration values along with template metadata so you can
    surface "Model X (CRPS-tuned, 12 lags, ERA5)" or similar in a UI.
    """
    configured_models_read = SessionWrapper(session=session).get_configured_models()

    # return
    return configured_models_read


@router.get(
    "/configured-models/{configuredModelId}",
    response_model=ConfiguredModelInfoRead,
    tags=["Models"],
    summary="View one configured model with its template",
)
@api_experimental
def get_configured_model_info(
    configured_model_id: Annotated[int, Path(alias="configuredModelId")],
    session: Session = Depends(get_session),
):
    """Look up a configured model together with the template it came from — the data you need to render a model detail pane (name, configuration values, covariates, version, ...).

    404 if the id is unknown.
    """
    configured_model = session.exec(
        select(ConfiguredModelDB)
        .where(ConfiguredModelDB.id == configured_model_id)
        .options(selectinload(ConfiguredModelDB.model_template))  # type: ignore[arg-type]
    ).first()
    if configured_model is None:
        raise HTTPException(status_code=404, detail="Configured model not found")
    return configured_model


@router.post(
    "/configured-models",
    response_model=ConfiguredModelDB,
    tags=["Models"],
    summary="Configure a template into a runnable model",
)
def add_configured_model(
    model_configuration: ModelConfigurationCreate,
    session: Session = Depends(get_session),
):
    """Bind a model template together with user-chosen option values into a new, named configured model — the unit that backtests and predictions actually reference.

    Use this when an operator has filled out the configuration form for a template
    (lags, precision, extra covariates, ...) and wants to save it. The new row
    inherits whether the template originated from a CHAPKit service. Returns 404 if
    the template id is unknown.
    """
    session_wrapper = SessionWrapper(session=session)
    model_template_id = model_configuration.model_template_id
    configuration_name = model_configuration.name
    # Inherit uses_chapkit from parent template so the model loads correctly at runtime
    template = session.exec(select(ModelTemplateDB).where(ModelTemplateDB.id == model_template_id)).first()
    if template is None:
        raise HTTPException(status_code=404, detail="Model template not found")
    uses_chapkit = template.uses_chapkit
    db_id = session_wrapper.add_configured_model(
        model_template_id,
        ModelConfiguration(
            user_option_values=model_configuration.user_option_values,
            additional_continuous_covariates=model_configuration.additional_continuous_covariates,
        ),
        configuration_name,
        uses_chapkit=uses_chapkit,
    )
    return session.get(ConfiguredModelDB, db_id)


@router.delete(
    "/configured-models/{configuredModelId}",
    tags=["Models"],
    summary="Retire a configured model",
)
async def delete_configured_model(
    configured_model_id: Annotated[int, Path(alias="configuredModelId")], session: Session = Depends(get_session)
):
    """Soft-delete a configured model so it stops showing up in pickers, while keeping the underlying row intact so historical backtests / predictions that reference it still resolve.

    The row stays in the database with ``archived=True``; existing references remain
    valid. 404 if the id is unknown.
    """
    configured_model = session.get(ConfiguredModelDB, configured_model_id)
    if configured_model is None:
        raise HTTPException(status_code=404, detail="Configured model not found")
    configured_model.archived = True
    session.add(configured_model)
    session.commit()
    return {"message": "deleted"}


###########
# prediction setups


@router.post(
    "/prediction-setups",
    response_model=DataBaseResponse,
    tags=["Prediction Setups"],
    summary="Promote a backtest into a reusable prediction config",
)
@api_experimental
async def create_prediction_setup(request: PredictionSetupCreate, session: Session = Depends(get_session)):
    """Save a backtest as a prediction setup — a named configuration you can rerun on fresh data, either ad-hoc via ``/run`` or on a cron schedule.

    Use this after evaluating a model on historical data and deciding it's good enough
    to operationalise. A backtest can back at most one setup (1-to-1 link). 404 if the
    backtest is missing, 409 if it already has a setup, 422 if the cron expression or
    quantile targets are malformed.
    """
    try:
        setup = prediction_setup_service.create_prediction_setup(
            session,
            backtest_id=request.backtest_id,
            name=request.name,
            schedule_cron_expression=request.schedule_cron_expression,
            schedule_enabled=request.schedule_enabled,
            quantile_targets=request.quantile_targets,
        )
    except prediction_setup_service.BacktestNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except prediction_setup_service.DuplicateSetupError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except prediction_setup_service.InvalidSetupError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    if setup.id is None:
        raise HTTPException(status_code=500, detail="PredictionSetup creation produced no id")
    return DataBaseResponse(id=setup.id)


@router.get(
    "/prediction-setups",
    response_model=list[PredictionSetupRead],
    tags=["Prediction Setups"],
    summary="Browse saved prediction setups",
)
@api_experimental
async def list_prediction_setups(session: Session = Depends(get_session)):
    """List every prediction setup so you can manage them, run them ad-hoc, or check which backtests have been promoted into a recurring forecast.

    Lightweight listing: each entry carries the setup's schedule, target backtest, and
    quantile config, but not the predictions it has produced. Fetch a single setup by
    id to get those.
    """
    return prediction_setup_service.list_prediction_setups(session)


@router.get(
    "/prediction-setups/{predictionSetupId}",
    response_model=PredictionSetupReadWithPredictions,
    tags=["Prediction Setups"],
    summary="View a prediction setup with its forecast history",
)
@api_experimental
async def get_prediction_setup(
    prediction_setup_id: Annotated[int, Path(alias="predictionSetupId")],
    session: Session = Depends(get_session),
):
    """Read a setup's configuration alongside every prediction it has produced, so a UI can show "what does this setup do" and "what has it actually forecast" on the same page.

    404 if the id is unknown.
    """
    try:
        return prediction_setup_service.get_prediction_setup(session, prediction_setup_id, include_predictions=True)
    except prediction_setup_service.PredictionSetupNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.patch(
    "/prediction-setups/{predictionSetupId}",
    response_model=PredictionSetupRead,
    tags=["Prediction Setups"],
    summary="Tweak a prediction setup's schedule or targets",
)
@api_experimental
async def update_prediction_setup(
    prediction_setup_id: Annotated[int, Path(alias="predictionSetupId")],
    request: PredictionSetupUpdate,
    session: Session = Depends(get_session),
):
    """Adjust a setup without recreating it — pause or resume the schedule, change the cron expression, swap quantile targets, rename it.

    Only the fields you actually send are touched, so partial updates are safe. 404 if
    the id is unknown, 422 if the new values are malformed.
    """
    update_data = request.model_dump(exclude_unset=True, by_alias=False)
    try:
        return prediction_setup_service.update_prediction_setup(session, prediction_setup_id, update_data)
    except prediction_setup_service.PredictionSetupNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except prediction_setup_service.InvalidSetupError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


def _cancel_jobs_for_prediction_setup(prediction_setup_id: int) -> None:
    """Cancel in-flight celery jobs tagged with this setup id and clear their Redis metadata.

    Called before deleting a PredictionSetup so a still-running job doesn't try to write
    `Prediction.prediction_setup_id=<deleted_id>` and fail the FK insert. Fails the delete
    with 503 if Redis is unreachable (we cannot guarantee the FK invariant otherwise); one
    stuck per-job cancel does not block the sweep.
    """
    try:
        keys: list[str] = redis.keys("job_meta:*")  # type: ignore[assignment]
    except Exception as e:
        logger.warning("Failed to sweep job metadata for prediction setup %d", prediction_setup_id, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Cannot delete PredictionSetup: job-metadata store unavailable, retry later",
        ) from e
    for key in keys:
        meta: dict[str, str] = redis.hgetall(key)  # type: ignore[assignment]
        if meta.get(PREDICTION_SETUP_ID_JOB_META_KEY) != str(prediction_setup_id):
            continue
        task_id = key.split(":", 1)[1]
        if meta.get("status", "").lower() in {"pending", "started", "running"}:
            try:
                worker.get_job(task_id).cancel()
            except Exception:
                logger.warning(
                    "Failed to cancel job %s for prediction setup %d", task_id, prediction_setup_id, exc_info=True
                )
        redis.delete(key)


@router.delete(
    "/prediction-setups/{predictionSetupId}",
    tags=["Prediction Setups"],
    summary="Retire a prediction setup",
)
@api_experimental
async def delete_prediction_setup(
    prediction_setup_id: Annotated[int, Path(alias="predictionSetupId")],
    session: Session = Depends(get_session),
):
    """Stop a setup from ever running again — cancels any in-flight jobs it has launched, then removes the setup row.

    Use this when a forecast workflow is being decommissioned. Cancellation of running
    jobs is essential here: a still-running job would otherwise try to write a
    foreign-key link to a deleted setup and crash. 404 if the id is unknown; 503 if
    Redis is unreachable (we can't safely cancel jobs in that case so the delete is
    refused).
    """
    # Verify existence first so a 404 doesn't waste a Redis sweep, then cancel in-flight
    # jobs BEFORE the DB delete — otherwise a still-running job would try to write
    # Prediction.prediction_setup_id=<deleted_id> and fail the FK insert.
    try:
        prediction_setup_service.get_prediction_setup(session, prediction_setup_id)
    except prediction_setup_service.PredictionSetupNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    _cancel_jobs_for_prediction_setup(prediction_setup_id)
    prediction_setup_service.delete_prediction_setup(session, prediction_setup_id)
    return {"message": "deleted"}


@router.post(
    "/prediction-setups/{predictionSetupId}/run",
    response_model=JobResponse,
    tags=["Prediction Setups"],
    summary="Run a prediction setup against fresh observations",
)
@api_experimental
async def run_prediction_setup(
    prediction_setup_id: Annotated[int, Path(alias="predictionSetupId")],
    request: RunPredictionSetupRequest,
    session: Session = Depends(get_session),
    database_url: str = Depends(get_database_url),
    worker_settings=Depends(get_settings),
):
    """Run a forecast from a saved setup using observations supplied directly in the request body — the manual equivalent of what the cron schedule does automatically.

    Use this when you want to forecast ahead of the schedule (a new data drop has
    arrived, a model investigation, etc.). Returns a job id; track it via
    ``/v1/jobs/{id}`` or filter the jobs list with ``predictionSetupId`` to see every
    job this setup has launched. Returns 404 if the setup is unknown, 409 if its
    configured model has been archived, 422 if ``provided_data`` is empty.
    """
    try:
        setup = prediction_setup_service.get_prediction_setup(session, prediction_setup_id)
    except prediction_setup_service.PredictionSetupNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    if setup.configured_model is None:
        # Defensive: the schema makes configured_model NOT NULL via FK, so this should not happen
        # outside of a manually-corrupted DB. Translating to a 500 keeps the contract honest.
        raise HTTPException(status_code=500, detail="PredictionSetup has no configured_model")
    if setup.configured_model.archived:
        raise HTTPException(
            status_code=409,
            detail=f"Configured model '{setup.configured_model.name}' is archived",
        )
    model_id = setup.configured_model.name

    if not request.provided_data:
        raise HTTPException(status_code=422, detail="provided_data cannot be empty")

    feature_names = list({entry.feature_name for entry in request.provided_data})
    dataclass = create_tsdataclass(feature_names)
    provided_data = observations_to_dataset(dataclass, request.provided_data, fill_missing=True)
    if "population" in feature_names:
        provided_data = provided_data.interpolate(["population"])
    provided_data, rejections = validate_full_dataset(feature_names, provided_data)
    if rejections:
        logger.warning(
            "%d observations rejected for prediction-setup %d",
            len(rejections),
            prediction_setup_id,
        )
    provided_data.set_polygons(FeatureCollectionModel.model_validate(request.geojson))

    # Normalize dataset type server-side, matching analytics.make_prediction. Whatever the
    # client sent in `request.type` (e.g. chap-scheduler defaults to "forecasting") gets
    # overridden so the persisted dataset is consistently tagged "prediction" and shows
    # up in prediction-filtered UI/queries. Use a local instead of mutating the request.
    dataset_type = "prediction"
    dataset_info = DataSetCreateInfo(name=request.name, type=dataset_type).model_dump()
    # Inherit the provider the setup's backtest was evaluated with, so a promoted
    # backtest predicts against the same future-weather source it was scored on.
    provider = setup.backtest.future_weather_provider
    if resolve_weather_provider(provider).leaks_future_data:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Prediction setup {prediction_setup_id} was evaluated with the '{provider}' future-weather "
                "provider, which reads the forecast window's own observations and so cannot forecast ahead. "
                "Re-run the backtest with a forecasting provider before running predictions from it."
            ),
        )
    prediction_params = PredictionParams(
        model_id=model_id, n_periods=request.n_periods, future_weather_provider=provider
    )
    job = worker.queue_db(
        wf.predict_pipeline_from_composite_dataset,
        feature_names,
        provided_data.model_dump(),
        request.name,
        dataset_create_info=dataset_info,
        prediction_params=prediction_params,
        prediction_setup_id=prediction_setup_id,
        configured_model_id=setup.configured_model_id,
        database_url=database_url,
        worker_config=worker_settings,
        **{JOB_TYPE_KW: JobType.PREDICTION, JOB_NAME_KW: request.name},
    )
    return JobResponse(id=job.id)
