from chap_core.climate_predictor import QuickForecastFetcher
from chap_core.database.database import SessionWrapper
from chap_core.datatypes import FullData, HealthPopulationData
from chap_core.predictor.model_registry import registry
from chap_core.assessment.prediction_evaluator import backtest as _backtest
from chap_core.rest_api_src.worker_functions import harmonize_health_dataset
from chap_core.data import DataSet as InMemoryDataSet

def run_backtest(estimator_id: registry.model_type, dataset_id: str, n_periods: int, n_splits: int, stride: int,
                 session: SessionWrapper):
    dataset = session.get_dataset(dataset_id, FullData)
    estimator = registry.get_model(estimator_id, ignore_env=True)
    predictions_list = _backtest(estimator,
                                 dataset,
                                 prediction_length=n_periods,
                                 n_test_sets=n_splits,
                                 stride=stride,
                                 weather_provider=QuickForecastFetcher)
    last_train_period = dataset.period_range[-1]
    db_id = session.add_evaluation_results(predictions_list, last_train_period, dataset_id, estimator_id)
    assert db_id is not None
    return db_id

def debug(session: SessionWrapper):
    return session.add_debug()

def harmonize_and_add_health_dataset(health_dataset: FullData, name: str, session: SessionWrapper) -> FullData:
    health_dataset = InMemoryDataSet.from_dict(health_dataset, HealthPopulationData)
    dataset = harmonize_health_dataset(health_dataset, usecwd_for_credentials=False)
    db_id = session.add_dataset(name, dataset, polygons=health_dataset.polygons.model_dump_json())
    return db_id