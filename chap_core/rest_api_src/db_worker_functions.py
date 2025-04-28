from typing import Optional

from chap_core.assessment.forecast import forecast_ahead
from chap_core.climate_predictor import QuickForecastFetcher
from chap_core.database.database import SessionWrapper
from chap_core.datatypes import FullData, HealthPopulationData, create_tsdataclass
from chap_core.predictor.model_registry import registry
from chap_core.assessment.prediction_evaluator import backtest as _backtest
from chap_core.rest_api_src.data_models import FetchRequest, BackTestCreate
# from chap_core.rest_api_src.v1.routers.crud import BackTestCreate
from chap_core.rest_api_src.worker_functions import harmonize_health_dataset, WorkerConfig
from chap_core.data import DataSet as InMemoryDataSet
from chap_core.time_period import Month


def trigger_exception(*args, **kwargs):
    raise Exception("Triggered exception")


def run_backtest(info: BackTestCreate,
                 n_periods: Optional[int] = None,
                 n_splits: int = 10,
                 stride: int = 1,
                 session: SessionWrapper = None):
    dataset = session.get_dataset(info.dataset_id, FullData)
    if n_periods is None:
        n_periods = _get_n_periods(dataset)
    estimator = registry.get_model(info.model_id, ignore_env=info.model_id.startswith('chap_ewars'))
    predictions_list = _backtest(estimator,
                                 dataset,
                                 prediction_length=n_periods,
                                 n_test_sets=n_splits,
                                 stride=stride,
                                 weather_provider=QuickForecastFetcher)
    last_train_period = dataset.period_range[-1]
    db_id = session.add_evaluation_results(predictions_list, last_train_period, info)
    assert db_id is not None
    return db_id


def run_prediction(estimator_id: registry.model_type, dataset_id: str, n_periods: Optional[int], name: str,
                   metadata: dict, session: SessionWrapper):
    dataset = session.get_dataset(dataset_id, FullData)
    if n_periods is None:
        n_periods = _get_n_periods(dataset)
    estimator = registry.get_model(estimator_id, ignore_env=estimator_id.startswith('chap_ewars'))
    predictions = forecast_ahead(estimator, dataset, n_periods)
    db_id = session.add_predictions(predictions, dataset_id, estimator_id, name, metadata)
    assert db_id is not None
    return db_id


def debug(session: SessionWrapper):
    return session.add_debug()


def harmonize_and_add_health_dataset(health_dataset: FullData, name: str, session: SessionWrapper,
                                     worker_config=WorkerConfig()) -> FullData:
    health_dataset = InMemoryDataSet.from_dict(health_dataset, HealthPopulationData)
    dataset = harmonize_health_dataset(health_dataset, usecwd_for_credentials=False, worker_config=worker_config)
    db_id = session.add_dataset(name, dataset, polygons=health_dataset.polygons.model_dump_json())
    return db_id


def harmonize_and_add_dataset(
        provided_field_names: list[str],
        data_to_be_fetched: list[FetchRequest],
        health_dataset: InMemoryDataSet,
        name: str,
        ds_type: str,
        session: SessionWrapper,
        worker_config=WorkerConfig()) -> FullData:
    provided_dataclass = create_tsdataclass(provided_field_names)
    health_dataset = InMemoryDataSet.from_dict(
        health_dataset, provided_dataclass)
    if len(data_to_be_fetched):
        full_dataset = harmonize_health_dataset(health_dataset,
                                                fetch_requests=data_to_be_fetched,
                                                usecwd_for_credentials=False,
                                                worker_config=worker_config)
    else:
        full_dataset = health_dataset
    db_id = session.add_dataset(name, full_dataset, polygons=health_dataset.polygons.model_dump_json(),
                                dataset_type=ds_type)
    return db_id


def _get_n_periods(health_dataset):
    frequency = 'M' if isinstance(health_dataset.period_range[0], Month) else 'W'
    n_periods = 3 if frequency == 'M' else 12
    return n_periods


def predict_pipeline_from_composite_dataset(provided_field_names: list[str],
                                            data_to_be_fetched: list[FetchRequest],
                                            health_dataset: InMemoryDataSet,
                                            name: str,
                                            model_id: registry.model_type,
                                            metadata: str,
                                            session: SessionWrapper,
                                            worker_config=WorkerConfig()) -> int:
    dataset_id = harmonize_and_add_dataset(provided_field_names, data_to_be_fetched, health_dataset, name,
                                           'prediction',
                                           session,
                                           worker_config)
    return run_prediction(model_id, dataset_id, None, name, metadata, session)
