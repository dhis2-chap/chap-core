"""Model evaluation through backtesting.

Provides functions for training a model and evaluating its predictions against
held-out test data using expanding window cross-validation. The main entry
points are ``backtest`` (yields per-split prediction results) and
``evaluate_model`` (runs a full evaluation with GluonTS metrics).
"""

import logging
from collections import defaultdict
from collections.abc import Iterable
from typing import Protocol, TypeVar

import numpy as np
import pandas as pd
from gluonts.evaluation import Evaluator
from gluonts.model import Forecast, SampleForecast
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from chap_core import get_temp_dir
from chap_core.assessment.dataset_splitting import (
    train_test_generator,
)
from chap_core.data.gluonts_adaptor.dataset import ForecastAdaptor
from chap_core.datatypes import Samples, SamplesWithTruth, TimeSeriesData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import PeriodRange

plt.set_loglevel(level="warning")
logger = logging.getLogger(__name__)


FeatureType = TypeVar("FeatureType", bound=TimeSeriesData)


class Predictor(Protocol):
    def predict(
        self,
        historic_data: DataSet[FeatureType],
        future_data: DataSet[TimeSeriesData],
    ) -> Samples: ...


class Estimator(Protocol):
    def train(self, data: DataSet) -> Predictor: ...


def backtest(
    estimator: Estimator, data: DataSet, prediction_length, n_test_sets, stride=1, weather_provider=None
) -> Iterable[DataSet]:
    """Train a model once and generate predictions for each test split.

    Uses ``train_test_generator`` to create an expanding window split of the
    data. The estimator is trained on the initial training set, then the
    trained predictor generates forecasts for each successive test window.

    Parameters
    ----------
    estimator
        Model estimator with a ``train`` method.
    data
        Full dataset to split and evaluate on.
    prediction_length
        Number of periods to predict per test window.
    n_test_sets
        Number of expanding window test splits.
    stride
        Periods to advance between successive splits.
    weather_provider
        Optional future weather data provider.

    Yields
    ------
    DataSet[SamplesWithTruth]
        For each test split, a dataset mapping locations to
        ``SamplesWithTruth`` (predicted samples merged with observed values).
    """
    train, test_generator = train_test_generator(
        data, prediction_length, n_test_sets, future_weather_provider=weather_provider
    )
    predictor = estimator.train(train)
    for historic_data, future_data, future_truth in test_generator:
        r = predictor.predict(historic_data, future_data)
        samples_with_truth = future_truth.merge(r, result_dataclass=SamplesWithTruth)  # type: ignore[arg-type]
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
    train, _test_generator = train_test_generator(
        data, prediction_length, n_test_sets, future_weather_provider=weather_provider
    )
    predictor = estimator.train(train)
    truth_data = {
        location: pd.DataFrame(
            data[location].disease_cases,
            index=data[location].time_period.to_period_index(),
        )
        for location in data.keys()  # noqa: SIM118
    }

    # transformed = create_multiloc_timeseries(truth_data)
    if report_filename is None:
        report_filename = str(get_temp_dir() / "evaluation_report.pdf")

    if report_filename is not None:
        logger.info(f"Plotting forecasts to {report_filename}")
        _, plot_test_generatro = train_test_generator(
            data, prediction_length, n_test_sets, future_weather_provider=weather_provider
        )
        forecasts_and_truths_generator = plot_forecasts(predictor, plot_test_generatro, truth_data, report_filename)

    logger.info("Getting forecasts")
    # forecast_list, tss = _get_forecast_generators(predictor, test_generator, truth_data)
    forecast_list, tss = zip(*forecasts_and_truths_generator, strict=False)

    logger.info("Evaluating")
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9], num_workers=None, allow_nan_forecast=True)
    results = evaluator(tss, forecast_list)
    logger.info("Finished Evaluating")
    return results


def create_multiloc_timeseries(truth_data):
    from chap_core.assessment.representations import MultiLocationDiseaseTimeSeries

    multi_location_disease_time_series = MultiLocationDiseaseTimeSeries()
    for location, df in truth_data.items():
        from chap_core.assessment.representations import DiseaseObservation, DiseaseTimeSeries

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
    truth_data: dict[str, pd.DataFrame],
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
        for location, samples in forecasts.items():  # type: ignore[attr-defined]
            forecast = ForecastAdaptor.from_samples(samples)
            t = truth_data[location]
            tss.append(t)
            forecast_list.append(forecast)

    return forecast_list, tss


def _get_forecast_dict(predictor: Predictor, test_generator) -> dict[str, list[SampleForecast]]:
    forecast_dict = defaultdict(list)

    for historic_data, future_data, _ in test_generator:
        assert len(future_data.period_range) > 0, (
            f"Future data must have at least one period {historic_data.period_range}, {future_data.period_range}"
        )
        forecasts = predictor.predict(historic_data, future_data)
        for location, samples in forecasts.items():  # type: ignore[attr-defined]
            forecast_dict[location].append(ForecastAdaptor.from_samples(samples))
    return forecast_dict


def plot_forecasts(predictor, test_instance, truth, pdf_filename):
    forecast_dict = _get_forecast_dict(predictor, test_instance)
    with PdfPages(pdf_filename) as pdf:
        for location, forecasts in forecast_dict.items():
            try:
                _t = truth[location]
            except KeyError:
                location = str(location)
                try:
                    _t = truth[location]
                except KeyError:
                    logger.error(f"Location {location!r} not found in truth data which has locations {truth.keys()}")
                    raise
                logging.warning(
                    f"Had to convert location to string {location}, something has maybe gone wrong at some point with data types"
                )

            for forecast in forecasts:
                # logging.info(forecasts)
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
        for location in truth.keys()  # noqa: SIM118
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


def generate_pdf_from_evaluation(evaluation, pdf_filename: str) -> None:
    """
    Generate old-style matplotlib PDF report from an Evaluation object.

    Creates a multi-page PDF with one page per location/split combination,
    showing historical observations and forecast distributions using GluonTS
    SampleForecast plotting.

    Args:
        evaluation: Evaluation object (from Evaluation.from_file or Evaluation.from_backtest)
        pdf_filename: Path to output PDF file
    """
    from chap_core.time_period import TimePeriod

    backtest = evaluation.to_backtest()
    flat_data = evaluation.to_flat()

    # Build observations dict from test period observations
    observations = backtest.dataset.observations
    obs_by_location: defaultdict[str, dict[str, float]] = defaultdict(dict)
    for obs in observations:
        if obs.feature_name == "disease_cases" and obs.value is not None:
            obs_by_location[obs.org_unit][obs.period] = obs.value

    # Add historical observations if available
    if flat_data.historical_observations is not None:
        historical_df = pd.DataFrame(flat_data.historical_observations)
        for _, row in historical_df.iterrows():
            location = row["location"]
            period = row["time_period"]
            value = row["disease_cases"]
            if value is not None and not np.isnan(value):
                obs_by_location[location][period] = value

    forecasts_by_loc_split: defaultdict[tuple[str, str], defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for fc in backtest.forecasts:
        key = (fc.org_unit, fc.last_seen_period)
        forecasts_by_loc_split[key][fc.period] = fc.values

    with PdfPages(pdf_filename) as pdf:
        for (location, last_seen_period), period_forecasts in sorted(forecasts_by_loc_split.items()):
            if location not in obs_by_location:
                logger.warning(f"No observations found for location {location}, skipping")
                continue

            obs_dict = obs_by_location[location]
            obs_df = pd.DataFrame(
                {"disease_cases": list(obs_dict.values())},
                index=pd.PeriodIndex([TimePeriod.parse(p).topandas() for p in obs_dict]),
            )
            obs_df = obs_df.sort_index()

            sorted_periods = sorted(period_forecasts.keys())
            samples_matrix = np.array([period_forecasts[p] for p in sorted_periods])

            forecast = ForecastAdaptor.from_samples(
                Samples(  # type: ignore[call-arg]
                    samples=samples_matrix,
                    time_period=PeriodRange.from_strings(sorted_periods),
                )
            )

            if np.any(np.isnan(forecast.samples)):
                logger.warning(f"Forecast for {location} at {last_seen_period} has NaN values")

            plt.figure(figsize=(8, 4))

            # Filter observations up to and including the forecast period
            obs_until_forecast_end = obs_df[obs_df.index <= forecast.index[-1]]

            forecast.plot(show_label=True)
            plt.plot(obs_until_forecast_end.to_timestamp(), label="Observed")

            plt.title(location)
            plt.legend()
            pdf.savefig()
            plt.close()

    logger.info(f"PDF report saved to {pdf_filename}")
