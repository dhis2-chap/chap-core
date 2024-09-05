from dataclasses import dataclass
from typing import Protocol, TypeVar

from gluonts.evaluation import Evaluator
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import plotly.express as px
from climate_health.assessment.dataset_splitting import get_split_points_for_data_set, split_test_train_on_period, \
    train_test_split, train_test_generator
from climate_health.assessment.multi_location_evaluator import MultiLocationEvaluator
from climate_health.data.gluonts_adaptor.dataset import ForecastAdaptor
from climate_health.datatypes import TimeSeriesData, Samples
from climate_health.predictor.naive_predictor import MultiRegionPoissonModel
from climate_health.reports import HTMLReport, HTMLSummaryReport
import logging

from climate_health.spatio_temporal_data.temporal_dataclass import DataSet

logger = logging.getLogger(__name__)


class AssessmentReport:
    def __init__(self, rmse_dict):
        self.rmse_dict = rmse_dict
        return


def make_assessment_report(prediction_dict, truth_dict, do_show=False) -> AssessmentReport:
    rmse_dict = {}
    for (prediction_key, prediction_value) in prediction_dict.items():
        rmse_dict[prediction_key] = root_mean_squared_error(list(truth_dict[prediction_key].values()),
                                                            list(prediction_value.values()))
    plot_rmse(rmse_dict, do_show=False)

    return AssessmentReport(rmse_dict)


def plot_rmse(rmse_dict, do_show=True):
    fig = px.line(x=list(rmse_dict.keys()),
                  y=list(rmse_dict.values()),
                  title='Root mean squared error per lag',
                  labels={'x': 'lag_ahead', 'y': 'RMSE'},
                  markers=True)
    if do_show:
        fig.show()
    return fig


def evaluate_model(data_set, external_model, max_splits=5, start_offset=20,
                   return_table=False, naive_model_cls=None, callback=None, mode='predict',
                   run_naive_predictor=True):
    '''
    Evaluate a model on a dataset using forecast cross validation
    '''
    if naive_model_cls is None:
        naive_model_cls = MultiRegionPoissonModel
    model_name = external_model.__class__.__name__
    naive_model_name = naive_model_cls.__name__
    evaluator = MultiLocationEvaluator(model_names=[model_name, naive_model_name], truth=data_set)
    split_points = get_split_points_for_data_set(data_set, max_splits=max_splits, start_offset=start_offset)
    logger.info(f'Split points: {split_points}')
    for (train_data, future_truth, future_climate_data) in split_test_train_on_period(data_set, split_points,
                                                                                      future_length=None,
                                                                                      include_future_weather=True):
        if hasattr(external_model, 'setup'):
            external_model.setup()
        external_model.train(train_data)
        predictions = getattr(external_model, mode)(future_climate_data)
        logger.info(f'Predictions: {predictions}')
        if callback:
            callback('predictions', predictions)
        evaluator.add_predictions(model_name, predictions)
        if run_naive_predictor:
            naive_predictor = naive_model_cls()
            naive_predictor.train(train_data)
            naive_predictions = getattr(naive_predictor, mode)(future_climate_data)
            evaluator.add_predictions(naive_model_name, naive_predictions)

        results: dict[str, pd.DataFrame] = evaluator.get_results()
    report_class = HTMLReport if mode == 'predict' else HTMLSummaryReport

    report_class.error_measure = 'mle'
    report = report_class.from_results(results)
    if return_table:
        for name, t in results.items():
            t['model'] = name
        results = pd.concat(results.values())
        return report, results
    return report


FetureType = TypeVar('FeatureType', bound=TimeSeriesData)


def without_disease(t):
    return t


class Predictor(Protocol):
    def predict(self, historic_data: DataSet[FetureType], future_data: DataSet[without_disease(FetureType)]) -> Samples:
        ...


class Estimator(Protocol):
    def train(self, data: DataSet) -> Predictor:
        ...


def evaluate_model(estimator: Estimator, data: DataSet, prediction_length=3, n_test_sets=4):
    train, test_generator = train_test_generator(data, prediction_length, n_test_sets)
    predictor = estimator.train(data)
    truth_data = {location: pd.DataFrame(data[location].disease_cases, index=data[location].time_period.to_period_index()) for location in data.keys()}
    tss = []
    forecast_list = []
    for historic_data, future_data, _ in test_generator:
        forecasts = predictor.predict(historic_data,future_data)
        for location, samples in forecasts.items():
            forecast = ForecastAdaptor.from_samples(samples)
            t = truth_data[location]
            tss.append(t)
            forecast_list.append(forecast)
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    results = evaluator(tss, forecast_list)
    return results
        #forecasts = ((predictor.predict(*test_pair[:2]), test_pair[2]) for test_pair in test_generator)