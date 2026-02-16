import inspect
import logging
from functools import wraps
from typing import get_type_hints

import numpy as np
from pydantic import BaseModel

from chap_core.api_types import BackTestParams
from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.forecast import forecast_ahead
from chap_core.assessment.prediction_evaluator import backtest as _backtest
from chap_core.climate_predictor import QuickForecastFetcher
from chap_core.data import DataSet as InMemoryDataSet
from chap_core.database.database import SessionWrapper
from chap_core.database.dataset_tables import DataSetCreateInfo
from chap_core.datatypes import HealthPopulationData, create_tsdataclass
from chap_core.log_config import get_status_logger
from chap_core.rest_api.data_models import BackTestCreate, FetchRequest, PredictionParams

# from chap_core.rest_api.v1.routers.crud import BackTestCreate
from chap_core.rest_api.worker_functions import WorkerConfig, harmonize_health_dataset
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import Month

logger = logging.getLogger(__name__)
status_logger = get_status_logger()


def convert_dicts_to_models(func):
    """Convert dict arguments to Pydantic models based on type hints."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Bind arguments to parameter names
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # Convert dicts to models where type hints indicate BaseModel
        for param_name, value in bound.arguments.items():
            if param_name in type_hints:
                expected_type = type_hints[param_name]
                # Check if it's a BaseModel subclass and value is a dict
                if isinstance(value, dict) and isinstance(expected_type, type) and issubclass(expected_type, BaseModel):
                    bound.arguments[param_name] = expected_type(**value)

        return func(*bound.args, **bound.kwargs)

    return wrapper


def trigger_exception(*args, **kwargs):
    raise Exception("Triggered exception")


def validate_and_filter_dataset_for_evaluation(
    dataset: DataSet, target_name: str, n_periods: int, n_splits: int, stride: int
) -> DataSet:
    evaluation_length = n_periods + (n_splits - 1) * stride
    new_data = {
        location: data
        for location, data in dataset.items()
        if np.any(np.logical_not(np.isnan(getattr(data, target_name)[:-evaluation_length])))
    }
    rejected: list[str] = []

    logger.warning(f"Rejected regions: {rejected} due to missing target values for the whole training period")
    logger.info(f"Remaining regions: {list(new_data.keys())} with {len(new_data)} entries")

    return DataSet(new_data, metadata=dataset.metadata, polygons=dataset.polygons)


# @convert_dicts_to_models
def run_backtest(
    info: BackTestCreate,
    n_periods: int | None = None,
    n_splits: int = 10,
    stride: int = 1,
    session: SessionWrapper | None = None,
):
    # NOTE: model_id arg from the user is actually the model's unique name identifier
    assert session is not None, "session is required"
    status_logger.info(f"Starting backtest for model '{info.model_id}' on dataset ID {info.dataset_id}")

    dataset = session.get_dataset(info.dataset_id)

    configured_model = session.get_configured_model_by_name(info.model_id)

    # hack to get who ewars model to work, it requires n_peridos=3.
    # todo: should be removed in future when system for model specific backtest params is implemented
    if configured_model.model_template.name == "ewars_plus":
        logger.warning("Forcing n_periods=3 for ewars_plus model")
        n_periods = 3

    if n_periods is None:
        n_periods = _get_n_periods(dataset)

    status_logger.info(f"Validating dataset with {len(list(dataset.locations()))} locations")
    dataset = validate_and_filter_dataset_for_evaluation(
        dataset,
        target_name="disease_cases",
        n_periods=n_periods,
        n_splits=n_splits,
        stride=stride,
    )

    status_logger.info(f"Running {n_splits} evaluation splits with prediction length {n_periods}")
    assert configured_model.id is not None, "configured_model.id is required"
    estimator = session.get_configured_model_with_code(configured_model.id)
    predictions_list = _backtest(
        estimator,
        dataset,
        prediction_length=n_periods,
        n_test_sets=n_splits,
        stride=stride,
        weather_provider=QuickForecastFetcher,
    )
    last_train_period = dataset.period_range[-1]
    evaluation = Evaluation.from_samples_with_truth(predictions_list, last_train_period, configured_model, info=info)
    backtest = evaluation.to_backtest()
    session.add_backtest(backtest)
    db_id = backtest.id
    assert db_id is not None
    status_logger.info(f"Backtest completed successfully. Results saved with ID {db_id}")
    return db_id


def run_prediction(
    model_id: str,
    dataset_id: str,
    n_periods: int | None,
    name: str,
    session: SessionWrapper,
):
    # NOTE: model_id arg from the user is actually the model's unique name identifier
    status_logger.info(f"Starting prediction for model '{model_id}' on dataset ID {dataset_id}")

    dataset = session.get_dataset(int(dataset_id))
    if n_periods is None:
        n_periods = _get_n_periods(dataset)

    status_logger.info(f"Training model and generating {n_periods} period forecast")
    configured_model = session.get_configured_model_by_name(model_id)
    assert configured_model.id is not None, "configured_model.id is required"
    estimator = session.get_configured_model_with_code(configured_model.id)
    predictions = forecast_ahead(estimator, dataset, n_periods)
    db_id = session.add_predictions(predictions, dataset_id, model_id, name)
    assert db_id is not None
    status_logger.info(f"Prediction completed successfully. Results saved with ID {db_id}")
    return db_id


def debug(session: SessionWrapper):
    return session.add_debug()


def harmonize_and_add_health_dataset(
    health_dataset: dict, name: str, session: SessionWrapper, worker_config: WorkerConfig = WorkerConfig()
) -> int:
    status_logger.info(f"Processing and adding dataset '{name}'")
    dataset_obj = InMemoryDataSet.from_dict(health_dataset, HealthPopulationData)  # type: ignore[arg-type]
    dataset = harmonize_health_dataset(dataset_obj, usecwd_for_credentials=False, worker_config=worker_config)
    db_id: int = session.add_dataset(
        DataSetCreateInfo(name=name), dataset, polygons=dataset_obj.polygons.model_dump_json()
    )
    status_logger.info(f"Dataset '{name}' added successfully with ID {db_id}")
    return db_id


def harmonize_and_add_dataset(
    provided_field_names: list[str],
    data_to_be_fetched: list[FetchRequest],
    health_dataset: dict,
    name: str,
    ds_type: str,
    session: SessionWrapper,
    worker_config: WorkerConfig = WorkerConfig(),
) -> int:
    status_logger.info(f"Processing and adding dataset '{name}' of type '{ds_type}'")
    provided_dataclass = create_tsdataclass(provided_field_names)
    dataset_obj = InMemoryDataSet.from_dict(health_dataset, provided_dataclass)
    if len(data_to_be_fetched):
        full_dataset = harmonize_health_dataset(
            dataset_obj, fetch_requests=data_to_be_fetched, usecwd_for_credentials=False, worker_config=worker_config
        )
    else:
        full_dataset = dataset_obj
    info = DataSetCreateInfo(name=name, type=ds_type)
    db_id: int = session.add_dataset(info, full_dataset, polygons=dataset_obj.polygons.model_dump_json())
    status_logger.info(f"Dataset '{name}' added successfully with ID {db_id}")
    return db_id


def _get_n_periods(health_dataset):
    frequency = "ME" if isinstance(health_dataset.period_range[0], Month) else "W"
    n_periods = 3 if frequency == "ME" else 12
    return n_periods


@convert_dicts_to_models
def predict_pipeline_from_composite_dataset(
    provided_field_names: list[str],
    health_dataset: dict,
    name: str,
    dataset_create_info: DataSetCreateInfo,
    prediction_params: PredictionParams,
    session: SessionWrapper,
    worker_config=WorkerConfig(),
) -> int:
    """
    This is the main pipeline function to run prediction from a dataset.
    """
    ds = InMemoryDataSet.from_dict(health_dataset, create_tsdataclass(provided_field_names))
    # dataset_info = DataSetCreateInfo.model_validate(dataset_create_info)

    dataset_id = session.add_dataset(
        dataset_info=dataset_create_info, orig_dataset=ds, polygons=ds.polygons.model_dump_json()
    )

    result: int = run_prediction(prediction_params.model_id, dataset_id, prediction_params.n_periods, name, session)
    return result


@convert_dicts_to_models
def run_backtest_from_dataset(
    feature_names: list[str],
    provided_data_model_dump: dict,
    backtest_name: str,
    model_id: str,
    dataset_info: DataSetCreateInfo,
    backtest_params: BackTestParams,
    session: SessionWrapper,
    worker_config=WorkerConfig(),
) -> int:
    ds = InMemoryDataSet.from_dict(provided_data_model_dump, create_tsdataclass(feature_names))
    dataset_id = session.add_dataset(dataset_info=dataset_info, orig_dataset=ds, polygons=ds.polygons.model_dump_json())
    backtest_create_info = BackTestCreate(name=backtest_name, dataset_id=dataset_id, model_id=model_id)
    if ds.frequency == "W" and backtest_params.stride < 4:
        logging.warning("Setting stride to 4 since its weekly data")
        backtest_params.stride = 4
    result: int = run_backtest(
        info=backtest_create_info,
        n_periods=backtest_params.n_periods,
        n_splits=backtest_params.n_splits,
        stride=backtest_params.stride,
        session=session,
    )
    return result
