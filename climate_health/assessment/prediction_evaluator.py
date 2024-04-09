from sklearn.metrics import root_mean_squared_error
import pandas as pd
import plotly.express as px
from climate_health.assessment.dataset_splitting import get_split_points_for_data_set, split_test_train_on_period
from climate_health.assessment.multi_location_evaluator import MultiLocationEvaluator
from climate_health.predictor.naive_predictor import MultiRegionPoissonModel
from climate_health.reports import HTMLReport
import logging

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


def evaluate_model(data_set, external_model, max_splits=5, start_offset=19, return_table=False, naive_model_cls=None, callback=None):
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
        predictions = external_model.predict(future_climate_data)
        logger.info(f'Predictions: {predictions}')
        if callback:
            callback('predictions', predictions)
        evaluator.add_predictions(model_name, predictions)
        naive_predictor = naive_model_cls()
        naive_predictor.train(train_data)
        naive_predictions = naive_predictor.predict(future_climate_data)
        evaluator.add_predictions(naive_model_name, naive_predictions)
    results = evaluator.get_results()
    HTMLReport.error_measure = 'mle'
    report = HTMLReport.from_results(results)
    if return_table:
        for name, t in results.items():
            t['model'] = name
        results = pd.concat(results.values())
        return report, results
    return report
