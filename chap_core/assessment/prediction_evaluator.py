from collections import defaultdict
from typing import Protocol, TypeVar, Iterable, Dict

from gluonts.evaluation import Evaluator
from gluonts.model import Forecast
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import plotly.express as px
from chap_core.assessment.dataset_splitting import (
    get_split_points_for_data_set,
    split_test_train_on_period,
    train_test_generator,
)
from chap_core.assessment.multi_location_evaluator import MultiLocationEvaluator
from chap_core.data.gluonts_adaptor.dataset import ForecastAdaptor
from chap_core.datatypes import TimeSeriesData, Samples
from chap_core.predictor.naive_predictor import MultiRegionPoissonModel
from chap_core._legacy.reports import HTMLReport, HTMLSummaryReport
import logging

from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

logger = logging.getLogger(__name__)


class AssessmentReport:
    def __init__(self, rmse_dict):
        self.rmse_dict = rmse_dict
        return


def make_assessment_report(prediction_dict, truth_dict, do_show=False) -> AssessmentReport:
    rmse_dict = {}
    for prediction_key, prediction_value in prediction_dict.items():
        rmse_dict[prediction_key] = root_mean_squared_error(
            list(truth_dict[prediction_key].values()), list(prediction_value.values())
        )
    plot_rmse(rmse_dict, do_show=False)

    return AssessmentReport(rmse_dict)


def plot_rmse(rmse_dict, do_show=True):
    fig = px.line(
        x=list(rmse_dict.keys()),
        y=list(rmse_dict.values()),
        title="Root mean squared error per lag",
        labels={"x": "lag_ahead", "y": "RMSE"},
        markers=True,
    )
    if do_show:
        fig.show()
    return fig


def _evaluate_model(
    data_set,
    external_model,
    max_splits=5,
    start_offset=20,
    return_table=False,
    naive_model_cls=None,
    callback=None,
    mode="predict",
    run_naive_predictor=True,
):
    """
    Evaluate a model on a dataset using forecast cross validation
    """
    if naive_model_cls is None:
        naive_model_cls = MultiRegionPoissonModel
    model_name = external_model.__class__.__name__
    naive_model_name = naive_model_cls.__name__
    evaluator = MultiLocationEvaluator(model_names=[model_name, naive_model_name], truth=data_set)
    split_points = get_split_points_for_data_set(data_set, max_splits=max_splits, start_offset=start_offset)
    logger.info(f"Split points: {split_points}")
    splitted_data = split_test_train_on_period(data_set, split_points, future_length=None, include_future_weather=True)

    for train_data, future_truth, future_climate_data in splitted_data:
        if hasattr(external_model, "setup"):
            external_model.setup()
        external_model.train(train_data)
        predictions = getattr(external_model, mode)(future_climate_data)
        logger.info(f"Predictions: {predictions}")
        if callback:
            callback("predictions", predictions)
        evaluator.add_predictions(model_name, predictions)
        if run_naive_predictor:
            naive_predictor = naive_model_cls()
            naive_predictor.train(train_data)
            naive_predictions = getattr(naive_predictor, mode)(future_climate_data)
            evaluator.add_predictions(naive_model_name, naive_predictions)

        results: dict[str, pd.DataFrame] = evaluator.get_results()
    report_class = HTMLReport if mode == "predict" else HTMLSummaryReport

    report_class.error_measure = "mle"
    report = report_class.from_results(results)
    if return_table:
        for name, t in results.items():
            t["model"] = name
        results = pd.concat(results.values())
        return report, results
    return report


FetureType = TypeVar("FeatureType", bound=TimeSeriesData)


def without_disease(t):
    return t


class Predictor(Protocol):
    def predict(
        self,
        historic_data: DataSet[FetureType],
        future_data: DataSet[without_disease(FetureType)],
    ) -> Samples: ...


class Estimator(Protocol):
    def train(self, data: DataSet) -> Predictor: ...


def backtest(estimator: Estimator,
    data: DataSet,
    prediction_length,
    n_test_sets, stride=1, weather_provider=None) -> Iterable[DataSet]:
    train, test_generator = train_test_generator(
        data, prediction_length, n_test_sets, future_weather_provider=weather_provider
    )
    predictor = estimator.train(train)
    for historic_data, future_data, _ in test_generator:
        yield predictor.predict(historic_data, future_data)


def relative_cases_mse(predicted: DataSet[Samples], truth: DataSet):
    ...
    



def evaluate_model(
    estimator: Estimator,
    data: DataSet,
    prediction_length=3,
    n_test_sets=4,
    report_filename=None,
    weather_provider=None,
):
    """
    Evaluate a model on a dataset on a held out test set, making multiple predictions on the test set
    using the same trained model

    Parameters
    ----------
    estimator : Estimator
        The estimator to train and evaluate
    data : DataSet
        The data to train and evaluate on
    prediction_length : int
        The number of periods to predict ahead
    n_test_sets : int
        The number of test sets to evaluate on

    Returns
    -------
    tuple
        Summary and individual evaluation results
    """
    train, test_generator = train_test_generator(
        data, prediction_length, n_test_sets, future_weather_provider=weather_provider
    )
    predictor = estimator.train(train)
    truth_data = {
        location: pd.DataFrame(
            data[location].disease_cases,
            index=data[location].time_period.to_period_index(),
        )
        for location in data.keys()
    }
    if report_filename is not None:
        _, plot_test_generatro = train_test_generator(
            data, prediction_length, n_test_sets, future_weather_provider=weather_provider
        )
        plot_forecasts(predictor, plot_test_generatro, truth_data, report_filename)
    forecast_list, tss = _get_forecast_generators(predictor, test_generator, truth_data)
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    results = evaluator(tss, forecast_list)
    return results


def evaluate_multi_model(
    estimator: Estimator,
    data: list[DataSet],
    prediction_length=3,
    n_test_sets=4,
    report_base_name=None,
):
    trains, test_geneartors = zip(*[train_test_generator(d, prediction_length, n_test_sets) for d in data])
    predictor = estimator.multi_train(trains)
    result_list = []
    for i, (data, test_generator) in enumerate(zip(data, test_geneartors)):
        truth_data = {
            location: pd.DataFrame(
                data[location].disease_cases,
                index=data[location].time_period.to_period_index(),
            )
            for location in data.keys()
        }
        if report_base_name is not None:
            _, plot_test_generatro = train_test_generator(data, prediction_length, n_test_sets)
            plot_forecasts(predictor, plot_test_generatro, truth_data, f"{report_base_name}_i.pdf")
        forecast_list, tss = _get_forecast_generators(predictor, test_generator, truth_data)
        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
        results = evaluator(tss, forecast_list)
        result_list.append(results)
    return results
    # forecasts = ((predictor.predict(*test_pair[:2]), test_pair[2]) for test_pair in test_generator)


def _get_forecast_generators(
    predictor: Predictor,
    test_generator: Iterable[tuple[DataSet, DataSet, DataSet]],
    truth_data: Dict[str, pd.DataFrame],
) -> tuple[list[Forecast], list[pd.DataFrame]]:
    """
    Get the forecast and truth data for a predictor and test generator.
    One entry is a combination of prediction start period and location

    Parameters
    ----------
    predictor : Predictor
        The predictor to evaluate
    test_generator : Iterable[tuple[DataSet, DataSet, DataSet]]
        The test generator to generate test data
    truth_data : dict[str, pd.DataFrame]
        The truth data for the locations
    """
    tss = []
    forecast_list = []
    for historic_data, future_data, _ in test_generator:
        forecasts = predictor.predict(historic_data, future_data)
        for location, samples in forecasts.items():
            forecast = ForecastAdaptor.from_samples(samples)
            t = truth_data[location]
            tss.append(t)
            forecast_list.append(forecast)

    return forecast_list, tss


def _get_forecast_dict(predictor: Predictor, test_generator) -> dict[str, list[Forecast]]:
    forecast_dict = defaultdict(list)

    for historic_data, future_data, _ in test_generator:
        assert (
            len(future_data.period_range) > 0
        ), f"Future data must have at least one period {historic_data.period_range}, {future_data.period_range}"
        forecasts = predictor.predict(historic_data, future_data)
        for location, samples in forecasts.items():
            forecast_dict[location].append(ForecastAdaptor.from_samples(samples))
    return forecast_dict


def plot_forecasts(predictor, test_instance, truth, pdf_filename):
    forecast_dict = _get_forecast_dict(predictor, test_instance)
    with PdfPages(pdf_filename) as pdf:
        for location, forecasts in forecast_dict.items():
            _t = truth[location]
            for forecast in forecasts:
                plt.figure(figsize=(8, 4))  # Set the figure size
                t = _t[_t.index <= forecast.index[-1]]
                forecast.plot(show_label=True)
                plotting_context = 52 * 6
                plt.plot(t[-plotting_context:].to_timestamp())
                plt.title(location)
                plt.legend()
                pdf.savefig()
                plt.close()  # Close the figure


def plot_predictions(predictions: DataSet[Samples], truth: DataSet, pdf_filename):
    truth_dict = {
        location: pd.DataFrame(
            truth[location].disease_cases,
            index=truth[location].time_period.to_period_index(),
        )
        for location in truth.keys()
    }
    with PdfPages(pdf_filename) as pdf:
        for location, prediction in predictions.items():
            prediction = ForecastAdaptor.from_samples(prediction)
            t = truth_dict[location]
            plt.figure(figsize=(8, 4))  # Set the figure size
            # t = _t[_t.index <= prediction.index[-1]]
            prediction.plot(show_label=True)
            context_length = 52 * 6
            plt.plot(t[-context_length:].to_timestamp())
            plt.title(location)
            plt.legend()
            pdf.savefig()
            plt.close()  # Close the figure


def plot_forecasts_list(predictor, test_instances, truth, pdf_filename):
    forecasts, tss = _get_forecast_generators(predictor, test_instances, truth)
    with PdfPages(pdf_filename) as pdf:
        for i, (forecast_entry, ts_entry) in enumerate(zip(forecasts, tss)):
            last_period = forecast_entry.index[-1]
            ts_entry = ts_entry[ts_entry.index <= last_period]
            plt.figure(figsize=(8, 4))  # Set the figure size
            plt.plot(ts_entry[-150:].to_timestamp())
            forecast_entry.plot(show_label=True)
            plt.title(str(i))
            plt.legend()
            pdf.savefig()
            plt.close()  # Close the figure

    return pdf_filename
