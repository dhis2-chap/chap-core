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
import logging
from functools import partial
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Path, Query, UploadFile
from sqlalchemy.orm import selectinload
from sqlmodel import Session, select
from starlette.responses import StreamingResponse

import chap_core.rest_api.db_worker_functions as wf
from chap_core.api_types import FeatureCollectionModel
from chap_core.data import DataSet as InMemoryDataSet
from chap_core.database.base_tables import DBModel
from chap_core.database.database import SessionWrapper
from chap_core.database.dataset_tables import (
    DataSet,
    DataSetCreateInfo,
    DataSetInfo,
    DataSetWithObservations,
    ObservationBase,
)
from chap_core.database.debug import DebugEntry
from chap_core.database.model_spec_tables import ModelSpecRead
from chap_core.database.model_templates_and_config_tables import (
    ConfiguredModelDB,
    ModelConfiguration,
    ModelTemplateDB,
    ModelTemplateInformation,
    ModelTemplateMetaData,
)
from chap_core.database.tables import BackTest, Prediction, PredictionInfo
from chap_core.datatypes import FullData, HealthPopulationData
from chap_core.geometry import Polygons
from chap_core.rest_api.celery_tasks import JOB_NAME_KW, JOB_TYPE_KW, CeleryPool, JobType
from chap_core.spatio_temporal_data.converters import observations_to_dataset

from ...data_models import BackTestCreate, BackTestRead, JobResponse
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
        from chap_core.models.external_chapkit_model import ml_service_info_to_model_template_config

        session_wrapper = SessionWrapper(session=session)
        for service in service_list.services:
            try:
                config = ml_service_info_to_model_template_config(service.info, service.url)
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

router_get = partial(router.get, response_model_by_alias=True)  # MAGIC!: This makes the endpoints return camelCase
worker: CeleryPool[Any] = CeleryPool()


###########
# backtests


@router.get("/backtests", response_model=list[BackTestRead], tags=["Backtests"])  # This should be called list
async def get_backtests(session: Session = Depends(get_session)):
    """
    Returns a list of backtests/evaluations with only the id and name
    """
    backtests = session.exec(
        select(BackTest).options(
            selectinload(BackTest.dataset).defer(DataSet.geojson),  # type: ignore[arg-type]
            selectinload(BackTest.configured_model).selectinload(ConfiguredModelDB.model_template),  # type: ignore[arg-type]
        )
    ).all()
    return backtests


@router_get("/backtests/{backtestId}/full", response_model=BackTest, tags=["Backtests"])
async def get_backtest(backtest_id: Annotated[int, Path(alias="backtestId")], session: Session = Depends(get_session)):
    backtest = session.get(BackTest, backtest_id)
    if backtest is None:
        raise HTTPException(status_code=404, detail="BackTest not found")
    return backtest


@router_get("/backtests/{backtestId}/info", response_model=BackTestRead, tags=["Backtests"])
def get_backtest_info(backtest_id: Annotated[int, Path(alias="backtestId")], session: Session = Depends(get_session)):
    backtest = session.exec(
        select(BackTest)
        .where(BackTest.id == backtest_id)
        .options(
            selectinload(BackTest.dataset).defer(DataSet.geojson),  # type: ignore[arg-type]
            selectinload(BackTest.configured_model).selectinload(ConfiguredModelDB.model_template),  # type: ignore[arg-type]
        )
    ).first()
    if backtest is None:
        raise HTTPException(status_code=404, detail="BackTest not found")
    return backtest


class BackTestUpdate(DBModel):
    name: str | None = None


@router.post("/backtests", response_model=JobResponse, tags=["Backtests"])
async def create_backtest(backtest: BackTestCreate, database_url: str = Depends(get_database_url)):
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
    backtest = session.get(BackTest, backtest_id)
    if backtest is None:
        raise HTTPException(status_code=404, detail="BackTest not found")
    session.delete(backtest)
    session.commit()
    return {"message": "deleted"}


@router.patch("/backtests/{backtestId}", response_model=BackTestRead, tags=["Backtests"])
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

    # Reload with eager loading to avoid lazy-load issues
    db_backtest = session.exec(
        select(BackTest)
        .where(BackTest.id == backtest_id)
        .options(
            selectinload(BackTest.dataset).defer(DataSet.geojson),  # type: ignore[arg-type]
            selectinload(BackTest.configured_model).selectinload(ConfiguredModelDB.model_template),  # type: ignore[arg-type]
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
        backtest = session.get(BackTest, backtest_id)
        if backtest is not None:
            session.delete(backtest)
            deleted_count += 1
    session.commit()
    return {"message": f"Deleted {deleted_count} backtests"}


###########
# predictions


class PredictionCreate(DBModel):
    dataset_id: int
    estimator_id: str
    n_periods: int


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


class DataBaseResponse(DBModel):
    id: int


class DatasetCreate(DataSetCreateInfo):
    observations: list[ObservationBase]
    geojson: FeatureCollectionModel


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
    # dataset = session.get(DataSet, dataset_id)
    # if dataset is None:
    #    raise HTTPException(status_code=404, detail="Dataset not found")
    sw = SessionWrapper(session=session)
    in_memory_dataset = sw.get_dataset(dataset_id)
    df = in_memory_dataset.to_pandas()
    # Convert time_period column to strings for proper serialization
    df["time_period"] = df["time_period"].astype(str)
    return df.to_dict(orient="records")


@router.get("/datasets/{datasetId}/csv", tags=["Datasets"])
async def get_dataset_csv(dataset_id: Annotated[int, Path(alias="datasetId")], session: Session = Depends(get_session)):
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


class ModelTemplateRead(DBModel, ModelTemplateInformation, ModelTemplateMetaData):
    """
    ModelTemplateRead is a read model for the ModelTemplateDB.
    It is used to return the model template in a readable format.
    """

    # TODO: should probably be moved somewhere else?
    name: str
    id: int
    user_options: dict | None = None
    required_covariates: list[str] = []
    version: str | None = None
    archived: bool = False
    health_status: str | None = None


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


class ModelConfigurationCreate(DBModel):
    name: str
    model_template_id: int
    user_option_values: dict | None = None
    additional_continuous_covariates: list[str] = []


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
    uses_chapkit = template.uses_chapkit if template else False
    db_id = session_wrapper.add_configured_model(
        model_template_id,
        ModelConfiguration(**model_configuration.model_dump()),
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
# models (alias for configured models)


@router.get("/models", response_model=list[ModelSpecRead], tags=["Models"])
def list_models(session: Session = Depends(get_session)):
    """List all models from the db (alias for configured models)"""
    return list_configured_models(session)


@router.post("/models", response_model=ConfiguredModelDB, tags=["Models"])
def add_model(
    model_configuration: ModelConfigurationCreate,
    session: Session = Depends(get_session),
):
    """Add a model to the database (alias for configured models)"""
    return add_configured_model(model_configuration, session)


#############
# other misc


@router.post("/debug", tags=["Debug"])
async def debug_entry(database_url: str = Depends(get_database_url)) -> JobResponse:
    job = worker.queue_db(wf.debug, database_url=database_url)
    return JobResponse(id=job.id)


@router.get("/debug/{debugId}", tags=["Debug"])
async def get_debug_entry(
    debug_id: Annotated[int, Path(alias="debugId")], session: Session = Depends(get_session)
) -> DebugEntry:
    debug = session.get(DebugEntry, debug_id)
    if debug is None:
        raise HTTPException(status_code=404, detail="Debug entry not found")
    return debug
