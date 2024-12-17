from chap_core.climate_predictor import QuickForecastFetcher
from chap_core.database.database import SessionWrapper
from chap_core.datatypes import FullData
from chap_core.predictor.model_registry import registry
from chap_core.assessment.prediction_evaluator import backtest as _backtest


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
    session.add_evaluation_results(predictions_list, last_train_period, dataset_id, estimator_id)


def debug(session: SessionWrapper):
    return session.add_debug()
