from collections import defaultdict
from typing import Protocol, TypeVar, Iterable, Dict
from gluonts.model import SampleForecast
from gluonts.evaluation import Evaluator
from gluonts.model import Forecast
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from chap_core.assessment.dataset_splitting import (
    train_test_generator,
)

from chap_core.data.gluonts_adaptor.dataset import ForecastAdaptor
from chap_core.datatypes import TimeSeriesData, Samples, SamplesWithTruth
import logging

from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

plt.set_loglevel(level="warning")
logger = logging.getLogger(__name__)


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


def backtest(
    estimator: Estimator, data: DataSet, prediction_length, n_test_sets, stride=1, weather_provider=None
) -> Iterable[DataSet]:
    train, test_generator = train_test_generator(
        data, prediction_length, n_test_sets, future_weather_provider=weather_provider
    )
    predictor = estimator.train(train)
    for historic_data, future_data, future_truth in test_generator:
        r = predictor.predict(historic_data, future_data)
        samples_with_truth = future_truth.merge(r, result_dataclass=SamplesWithTruth)
        yield samples_with_truth


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
    logger.info(f"Evaluating {estimator} with {n_test_sets} test sets for {prediction_length} periods ahead")
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

    # transformed = create_multiloc_timeseries(truth_data)
    if report_filename is None:
        report_filename = "evaluation_report.pdf"

    if report_filename is not None:
        logger.info(f"Plotting forecasts to {report_filename}")
        _, plot_test_generatro = train_test_generator(
            data, prediction_length, n_test_sets, future_weather_provider=weather_provider
        )
        forecasts_and_truths_generator = plot_forecasts(predictor, plot_test_generatro, truth_data, report_filename)

    logger.info("Getting forecasts")
    # forecast_list, tss = _get_forecast_generators(predictor, test_generator, truth_data)
    forecast_list, tss = zip(*forecasts_and_truths_generator)

    logger.info("Evaluating")
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9], num_workers=None, allow_nan_forecast=True)
    results = evaluator(tss, forecast_list)
    logger.info("Finished Evaluating")
    return results


def create_multiloc_timeseries(truth_data):
    from chap_core.assessment.representations import MultiLocationDiseaseTimeSeries

    multi_location_disease_time_series = MultiLocationDiseaseTimeSeries()
    for location, df in truth_data.items():
        from chap_core.assessment.representations import DiseaseTimeSeries
        from chap_core.assessment.representations import DiseaseObservation

        disease_time_series = DiseaseTimeSeries(
            observations=[
                DiseaseObservation(time_period=period, disease_cases=cases)
                for period, cases in df.itertuples(index=True, name="Pandas")
            ]
        )
        multi_location_disease_time_series[location] = disease_time_series
    return multi_location_disease_time_series


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


def _get_forecast_dict(predictor: Predictor, test_generator) -> dict[str, list[SampleForecast]]:
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
            logging.info(f"Running on location {location}")
            try:
                _t = truth[location]
            except KeyError:
                location = str(location)
                try:
                    _t = truth[location]
                except KeyError:
                    logger.error(
                        f"Location {repr(location)} not found in truth data which has locations {truth.keys()}"
                    )
                    raise
                logging.warning(
                    f"Had to convert location to string {location}, something has maybe gone wrong at some point with data types"
                )

            for forecast in forecasts:
                logging.info("Forecasts: ")
                logging.info(forecasts)
                if np.any(np.isnan(forecast.samples)):
                    logger.warning(f"Forecast {forecast} has NaN values: {forecast.samples}")

                plt.figure(figsize=(8, 4))  # Set the figure size
                t = _t[_t.index <= forecast.index[-1]]
                forecast.plot(show_label=True)
                plotting_context = 52 * 6
                plt.plot(t[-plotting_context:].to_timestamp())
                plt.title(location)
                plt.legend()
                pdf.savefig()
                plt.close()  # Close the figure
                yield forecast, t


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
