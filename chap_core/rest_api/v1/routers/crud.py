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
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Path, Query, UploadFile
from sqlalchemy.orm import selectinload
from sqlmodel import Session, select
from starlette.responses import StreamingResponse

import chap_core.rest_api.db_worker_functions as wf
from chap_core.api_types import FeatureCollectionModel
from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.metrics import compute_all_detailed_metrics
from chap_core.data import DataSet as InMemoryDataSet
from chap_core.database.database import SessionWrapper
from chap_core.database.dataset_tables import (
    DataSet,
    DataSetCreateInfo,
    DataSetInfo,
    DataSetWithObservations,
)
from chap_core.database.model_spec_tables import ModelSpecRead
from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB, ModelConfiguration, ModelTemplateDB
from chap_core.database.tables import (
    Backtest,
    Prediction,
    PredictionInfo,
    PredictionSetupRead,
    PredictionSetupReadWithPredictions,
)
from chap_core.datatypes import FullData, HealthPopulationData, create_tsdataclass
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
from chap_core.services import prediction_setup_service
from chap_core.spatio_temporal_data.converters import observations_to_dataset

from ...data_models import (
    BacktestCreate,
    BacktestRead,
    BacktestUpdate,
    ConfiguredModelInfoRead,
    DataBaseResponse,
    DatasetCreate,
    JobResponse,
    ModelConfigurationCreate,
    ModelTemplateRead,
    PredictionCreate,
    PredictionParams,
    PredictionSetupCreate,
    PredictionSetupUpdate,
    RunPredictionSetupRequest,
)
from .analytics import validate_full_dataset
from .dependencies import get_database_url, get_session, get_settings

logger = logging.getLogger(__name__)


def _sync_live_chapkit_services(session: Session) -> set[str]:
    """Sync live chapkit services from the v2 registry into the DB.

    Queries the Redis-backed Orchestrator for registered services and
    upserts each as a model template. For new templates (no configured
    models yet), also fetches configs from the chapkit service and
    creates configured models. Silently skips if Redis is unavailable.

    Returns the set of live service IDs from the registry.
    """
    try:
        from chap_core.rest_api.v2.dependencies import get_orchestrator

        orchestrator = get_orchestrator()
        service_list = orchestrator.get_all()
    except Exception:
        logger.debug("Could not reach service registry, skipping chapkit sync")
        return set()

    live_ids = {s.info.id for s in service_list.services}

    if service_list.count > 0:
        from chap_core.models.chapkit_rest_api_wrapper import CHAPKitRestAPIWrapper
        from chap_core.models.external_chapkit_model import (
            _parse_user_options_from_config_schema,
            ml_service_info_to_model_template_config,
        )

        session_wrapper = SessionWrapper(session=session)
        for service in service_list.services:
            try:
                # Fetch the config schema so user_options (n_lags, precision, etc.)
                # are populated on the template row — both on first discovery and
                # on every subsequent sync so schema changes propagate without a
                # DB wipe.
                user_options: dict = {}
                try:
                    client = CHAPKitRestAPIWrapper(service.url, timeout=5)
                    schema = client.get_config_schema()
                    user_options = _parse_user_options_from_config_schema(schema)
                except Exception:
                    logger.debug("Could not fetch config schema from %s", service.url)

                config = ml_service_info_to_model_template_config(service.info, service.url, user_options)
                template_id = session_wrapper.add_model_template_from_yaml_config(config)
                # Mark template as chapkit-originated for archival tracking
                template = session.exec(select(ModelTemplateDB).where(ModelTemplateDB.id == template_id)).one()
                template.uses_chapkit = True
                session.commit()
                _sync_chapkit_configured_models(session_wrapper, template_id, service.url, CHAPKitRestAPIWrapper)
            except Exception:
                logger.warning("Failed to sync chapkit service %s", service.id, exc_info=True)

    _archive_stale_chapkit_templates(session, service_list)
    return live_ids


def _archive_stale_chapkit_templates(session: Session, service_list) -> None:
    """Archive chapkit templates whose services are no longer live."""
    live_names = {s.info.id for s in service_list.services}
    chapkit_templates = session.exec(
        select(ModelTemplateDB).where(
            ModelTemplateDB.uses_chapkit == True,
            ModelTemplateDB.archived == False,
        )
    ).all()
    for template in chapkit_templates:
        if template.name not in live_names:
            template.archived = True
            session.add(template)
    session.commit()


def _resolve_chapkit_default_additional_covariates(client) -> list[str]:
    """Probe a chapkit service for its BaseConfig `additional_continuous_covariates` default.

    Chapkit's `/api/v1/configs/$schema` endpoint doesn't expose `default_factory`
    values, so the only way to learn what the service considers a sensible default
    covariate set is to materialize a config with an empty `data` dict and read
    the pydantic-populated result back. We then delete the probe config to avoid
    leaving cruft in the service's DB.

    Returns an empty list on any failure — callers should treat the missing
    defaults as "no additional covariates".
    """
    import time

    probe_name = f"__chap_probe_defaults_{int(time.time() * 1000000)}__"
    try:
        probe = client.create_config({"name": probe_name, "data": {}})
    except Exception:
        logger.debug("Chapkit default probe POST failed", exc_info=True)
        return []

    try:
        data = getattr(probe, "data", None) or {}
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        result = list(data.get("additional_continuous_covariates", []) or [])
    except Exception:
        logger.debug("Chapkit default probe response parse failed", exc_info=True)
        result = []

    client.delete_config(str(probe.id))
    return result


def _sync_chapkit_configured_models(
    session_wrapper: SessionWrapper,
    template_id: int,
    service_url: str,
    wrapper_cls: type,
) -> None:
    """Sync configured models from a chapkit service into the DB.

    Skips if the template already has configured models (only syncs
    on first discovery). Creates a default configured model if the
    service has no configs.

    `additional_continuous_covariates` for each configured model is seeded
    from the chapkit service's `BaseConfig` defaults via a one-time probe
    against `/api/v1/configs`. This matches the legacy config-file-driven
    behaviour where the overlay YAML supplied the same field, and it is
    what the modeling app reads to render the model card's covariate count
    and the data-mapping dialog slots.
    """
    existing = session_wrapper.session.exec(
        select(ConfiguredModelDB).where(ConfiguredModelDB.model_template_id == template_id)
    ).first()
    if existing is not None:
        return

    client = wrapper_cls(service_url, timeout=5)
    try:
        configs = client.list_configs()
    except Exception:
        logger.debug("Could not fetch configs from %s, will retry next sync", service_url)
        return

    default_additional = _resolve_chapkit_default_additional_covariates(client)

    if not configs:
        session_wrapper.add_configured_model(
            template_id,
            ModelConfiguration(user_option_values={}, additional_continuous_covariates=default_additional),
            "default",
            uses_chapkit=True,
        )
        return

    for cfg in configs:
        # Chapkit manages its own config data; chap-core stores the
        # configured model as a reference only, with empty user options.
        # Carry over `additional_continuous_covariates` from the config's
        # own data when present; otherwise fall back to the service-level
        # default probed above.
        cfg_data = getattr(cfg, "data", None) or {}
        if hasattr(cfg_data, "model_dump"):
            cfg_data = cfg_data.model_dump()
        cfg_additional = list(cfg_data.get("additional_continuous_covariates", []) or []) or default_additional
        session_wrapper.add_configured_model(
            template_id,
            ModelConfiguration(user_option_values={}, additional_continuous_covariates=cfg_additional),
            cfg.name,
            uses_chapkit=True,
        )


router = APIRouter(prefix="/crud")

worker: CeleryPool[Any] = CeleryPool()


###########
# backtests


@router.get("/backtests", response_model=list[BacktestRead], tags=["Backtests"])  # This should be called list
async def get_backtests(session: Session = Depends(get_session)):
    """
    Returns a list of backtests/evaluations with only the id and name
    """
    backtests = session.exec(
        select(Backtest).options(
            selectinload(Backtest.dataset).defer(DataSet.geojson),  # type: ignore[arg-type]
            selectinload(Backtest.configured_model).selectinload(ConfiguredModelDB.model_template),  # type: ignore[arg-type]
            selectinload(Backtest.prediction_setup),  # type: ignore[arg-type]
        )
    ).all()
    return backtests


@router.get("/backtests/{backtestId}/full", response_model=Backtest, tags=["Backtests"])
async def get_backtest(backtest_id: Annotated[int, Path(alias="backtestId")], session: Session = Depends(get_session)):
    backtest = session.get(Backtest, backtest_id)
    if backtest is None:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return backtest


@router.get("/backtests/{backtestId}/info", response_model=BacktestRead, tags=["Backtests"])
@router.get("/backtests/{backtestId}", response_model=BacktestRead, tags=["Backtests"])
def get_backtest_info(backtest_id: Annotated[int, Path(alias="backtestId")], session: Session = Depends(get_session)):
    backtest = session.exec(
        select(Backtest)
        .where(Backtest.id == backtest_id)
        .options(
            selectinload(Backtest.dataset).defer(DataSet.geojson),  # type: ignore[arg-type]
            selectinload(Backtest.configured_model).selectinload(ConfiguredModelDB.model_template),  # type: ignore[arg-type]
            selectinload(Backtest.prediction_setup),  # type: ignore[arg-type]
        )
    ).first()
    if backtest is None:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return backtest


@router.get("/metric/csv", tags=["Metrics"])
async def get_metrics_csv(
    backtest_id: Annotated[int, Query(alias="backtestId")],
    session: Session = Depends(get_session),
):
    """
    Download per-location / per-time_period / per-horizon metric values as a
    long-format CSV. Every applicable metric in
    `chap_core.assessment.metrics.available_metrics` is included as rows.

    Currently takes a single backtest via the `backtestId` query parameter; the
    path is scoped to `/metric/` so it can be extended to accept multiple
    evaluations later without a breaking change.
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


@router.post("/backtests", response_model=JobResponse, tags=["Backtests"])
async def create_backtest(
    backtest: BacktestCreate,
    database_url: str = Depends(get_database_url),
    session: Session = Depends(get_session),
):
    # `BacktestCreate.model_id` accepts either the configured-model name
    # (what the DB column actually stores) or the integer primary key (what
    # most API clients reach for because that's what GET /v1/crud/configured-models
    # returns). The worker's run_backtest() normalises int -> name through
    # `SessionWrapper.get_configured_model_by_id_or_name` before touching
    # anything else, so the endpoint itself stays dumb and there's exactly
    # one resolution point.
    if session.get(DataSet, backtest.dataset_id) is None:
        raise HTTPException(status_code=404, detail=f"Dataset {backtest.dataset_id} not found")
    job = worker.queue_db(
        wf.run_backtest,
        backtest,
        database_url=database_url,
        **{JOB_TYPE_KW: JobType.EVALUATION_LEGACY, JOB_NAME_KW: backtest.name},
    )

    return JobResponse(id=job.id)


@router.delete("/backtests/{backtestId}", tags=["Backtests"])
async def delete_backtest(
    backtest_id: Annotated[int, Path(alias="backtestId")], session: Session = Depends(get_session)
):
    backtest = session.get(Backtest, backtest_id)
    if backtest is None:
        raise HTTPException(status_code=404, detail="Backtest not found")
    session.delete(backtest)
    session.commit()
    return {"message": "deleted"}


@router.patch("/backtests/{backtestId}", response_model=BacktestRead, tags=["Backtests"])
async def update_backtest(
    backtest_id: Annotated[int, Path(alias="backtestId")],
    backtest_update: BacktestUpdate,
    session: Session = Depends(get_session),
):
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
            selectinload(Backtest.dataset).defer(DataSet.geojson),  # type: ignore[arg-type]
            selectinload(Backtest.configured_model).selectinload(ConfiguredModelDB.model_template),  # type: ignore[arg-type]
            selectinload(Backtest.prediction_setup),  # type: ignore[arg-type]
        )
    ).first()
    return db_backtest


@router.delete("/backtests", tags=["Backtests"])
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


@router.get("/predictions", response_model=list[PredictionInfo], tags=["Predictions"])
async def get_predictions(session: Session = Depends(get_session)):
    session_wrapper = SessionWrapper(session=session)
    return [
        prediction for prediction in session_wrapper.list_all(Prediction) if prediction.configured_model is not None
    ]


@router.get("/predictions/{predictionId}", response_model=PredictionInfo, tags=["Predictions"])
async def get_prediction(
    prediction_id: Annotated[int, Path(alias="predictionId")], session: Session = Depends(get_session)
):
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return prediction


@router.post("/predictions", response_model=JobResponse, tags=["Predictions"])
async def create_prediction(prediction: PredictionCreate):
    raise HTTPException(status_code=501, detail="Not implemented")


@router.delete("/predictions/{predictionId}", tags=["Predictions"])
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


@router.get("/datasets", response_model=list[DataSetInfo], tags=["Datasets"])
async def get_datasets(session: Session = Depends(get_session)):
    datasets = session.exec(select(DataSet)).all()
    return datasets


@router.get("/datasets/{datasetId}", response_model=DataSetWithObservations, tags=["Datasets"])
async def get_dataset(dataset_id: Annotated[int, Path(alias="datasetId")], session: Session = Depends(get_session)):
    # dataset = session.exec(select(DataSet).where(DataSet.id == dataset_id)).first()
    dataset = session.get(DataSet, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    assert len(dataset.observations) > 0
    for obs in dataset.observations:
        obs.value = obs.value if obs.value is None or np.isfinite(obs.value) else None
    return dataset


@router.post("/datasets", tags=["Datasets"])
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


@router.post("/datasets/csvFile", tags=["Datasets"])
async def create_dataset_csv(
    csv_file: UploadFile = File(...),
    geojson_file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> DataBaseResponse:
    import io

    csv_content = await csv_file.read()
    dataset = InMemoryDataSet.from_csv(io.BytesIO(csv_content), dataclass=FullData)
    geo_json_content = await geojson_file.read()
    features = Polygons.from_geojson(json.loads(geo_json_content), id_property="NAME_1").feature_collection()
    dataset_id = SessionWrapper(session=session).add_dataset(
        DataSetCreateInfo(name="csv_file"), dataset, features.model_dump_json()
    )
    return DataBaseResponse(id=dataset_id)


@router.get("/datasets/{datasetId}/df", tags=["Datasets"])
async def get_dataset_df(dataset_id: Annotated[int, Path(alias="datasetId")], session: Session = Depends(get_session)):
    if session.get(DataSet, dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    sw = SessionWrapper(session=session)
    in_memory_dataset = sw.get_dataset(dataset_id)
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


@router.get("/datasets/{datasetId}/csv", tags=["Datasets"])
async def get_dataset_csv(dataset_id: Annotated[int, Path(alias="datasetId")], session: Session = Depends(get_session)):
    if session.get(DataSet, dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    sw = SessionWrapper(session=session)
    in_memory_dataset = sw.get_dataset(dataset_id)
    df = in_memory_dataset.to_pandas()
    df["time_period"] = df["time_period"].astype(str)

    csv_content = df.to_csv(index=False)
    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=dataset_{dataset_id}.csv"},
    )


@router.delete("/datasets/{datasetId}", tags=["Datasets"])
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


@router.get("/model-templates", response_model=list[ModelTemplateRead], tags=["Models"])
async def list_model_templates(session: Session = Depends(get_session)):
    """
    Lists all model templates from the db, including archived.
    Also syncs live chapkit services from the v2 service registry
    into the database (upsert by name).
    """
    live_ids = _sync_live_chapkit_services(session)
    model_templates = session.exec(select(ModelTemplateDB)).all()

    results = []
    for t in model_templates:
        read = ModelTemplateRead.model_validate(t)
        if t.name in live_ids:
            read.health_status = "live"
        results.append(read)
    return results


###########
# configured models


@router.get("/configured-models", response_model=list[ModelSpecRead], tags=["Models"])
def list_configured_models(session: Session = Depends(get_session)):
    """List all configured models from the db"""
    configured_models_read = SessionWrapper(session=session).get_configured_models()

    # return
    return configured_models_read


@router.get(
    "/configured-models/{configuredModelId}",
    response_model=ConfiguredModelInfoRead,
    tags=["Models"],
)
@api_experimental
def get_configured_model_info(
    configured_model_id: Annotated[int, Path(alias="configuredModelId")],
    session: Session = Depends(get_session),
):
    """Return the detail view for a single configured model, including its template."""
    configured_model = session.exec(
        select(ConfiguredModelDB)
        .where(ConfiguredModelDB.id == configured_model_id)
        .options(selectinload(ConfiguredModelDB.model_template))  # type: ignore[arg-type]
    ).first()
    if configured_model is None:
        raise HTTPException(status_code=404, detail="Configured model not found")
    return configured_model


@router.post("/configured-models", response_model=ConfiguredModelDB, tags=["Models"])
def add_configured_model(
    model_configuration: ModelConfigurationCreate,
    session: Session = Depends(get_session),
):
    """Add a configured model to the database"""
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


@router.delete("/configured-models/{configuredModelId}", tags=["Models"])
async def delete_configured_model(
    configured_model_id: Annotated[int, Path(alias="configuredModelId")], session: Session = Depends(get_session)
):
    """Soft delete a configured model by setting archived to True"""
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
)
@api_experimental
async def create_prediction_setup(request: PredictionSetupCreate, session: Session = Depends(get_session)):
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
)
@api_experimental
async def list_prediction_setups(session: Session = Depends(get_session)):
    return prediction_setup_service.list_prediction_setups(session)


@router.get(
    "/prediction-setups/{predictionSetupId}",
    response_model=PredictionSetupReadWithPredictions,
    tags=["Prediction Setups"],
)
@api_experimental
async def get_prediction_setup(
    prediction_setup_id: Annotated[int, Path(alias="predictionSetupId")],
    session: Session = Depends(get_session),
):
    try:
        return prediction_setup_service.get_prediction_setup(session, prediction_setup_id, include_predictions=True)
    except prediction_setup_service.PredictionSetupNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.patch(
    "/prediction-setups/{predictionSetupId}",
    response_model=PredictionSetupRead,
    tags=["Prediction Setups"],
)
@api_experimental
async def update_prediction_setup(
    prediction_setup_id: Annotated[int, Path(alias="predictionSetupId")],
    request: PredictionSetupUpdate,
    session: Session = Depends(get_session),
):
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


@router.delete("/prediction-setups/{predictionSetupId}", tags=["Prediction Setups"])
@api_experimental
async def delete_prediction_setup(
    prediction_setup_id: Annotated[int, Path(alias="predictionSetupId")],
    session: Session = Depends(get_session),
):
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
)
@api_experimental
async def run_prediction_setup(
    prediction_setup_id: Annotated[int, Path(alias="predictionSetupId")],
    request: RunPredictionSetupRequest,
    session: Session = Depends(get_session),
    database_url: str = Depends(get_database_url),
    worker_settings=Depends(get_settings),
):
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
    prediction_params = PredictionParams(model_id=model_id, n_periods=request.n_periods)
    job = worker.queue_db(
        wf.predict_pipeline_from_composite_dataset,
        feature_names,
        provided_data.model_dump(),
        request.name,
        dataset_create_info=dataset_info,
        prediction_params=prediction_params,
        prediction_setup_id=prediction_setup_id,
        database_url=database_url,
        worker_config=worker_settings,
        **{JOB_TYPE_KW: JobType.PREDICTION, JOB_NAME_KW: request.name},
    )
    return JobResponse(id=job.id)
