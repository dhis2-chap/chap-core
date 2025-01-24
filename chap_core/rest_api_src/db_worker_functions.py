from chap_core.assessment.forecast import forecast_ahead
from chap_core.climate_predictor import QuickForecastFetcher
from chap_core.database.database import SessionWrapper
from chap_core.datatypes import FullData, HealthPopulationData
from chap_core.predictor.model_registry import registry
from chap_core.assessment.prediction_evaluator import backtest as _backtest
from chap_core.rest_api_src.data_models import DatasetMakeRequest
from chap_core.rest_api_src.worker_functions import harmonize_health_dataset, WorkerConfig
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


def run_prediction(estimator_id: registry.model_type, dataset_id: str, n_periods: int, session: SessionWrapper):
    dataset = session.get_dataset(dataset_id, FullData)
    estimator = registry.get_model(estimator_id, ignore_env=True)
    predictions = forecast_ahead(estimator, dataset, n_periods)
    db_id = session.add_predictions(predictions, dataset_id, estimator_id)
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

def harmonize_and_add_composite_dataset(
        health_dataset: InMemoryDataSet[HealthPopulationData],
        request: DatasetMakeRequest,
        name: str,
        session: SessionWrapper,
        worker_config=WorkerConfig()) -> FullData:
    raise NotImplementedError('This function is not implemented yet')



def predict_pipeline_from_health_dataset(health_dataset: HealthPopulationData,
                                         name: str, model_id: registry.model_type, session: SessionWrapper,
                                         worker_config=WorkerConfig()):
    dataset_id = harmonize_and_add_health_dataset(health_dataset, name, session, worker_config)
    return run_prediction(model_id, dataset_id, 3, session)
    #return run_backtest(model_id, dataset_id, 3, 4, 1, session)
