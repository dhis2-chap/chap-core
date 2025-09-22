import itertools
from collections import defaultdict
from typing import Dict, List

from chap_core.assessment.evaluator import Evaluator
from chap_core.assessment.representations import (
    MultiLocationForecast,
    Samples,
    Forecast,
    MultiLocationDiseaseTimeSeries,
    DiseaseObservation,
    DiseaseTimeSeries,
    MultiLocationErrorTimeSeries,
    ErrorTimeSeries,
    Error,
)
from chap_core.database.tables import BackTestForecast
from chap_core.database.dataset_tables import ObservationBase


def convert_to_multi_location_forecast(backTestList: List[BackTestForecast]) -> Dict[str, MultiLocationForecast]:
    # Group samples by location
    all_splitpoint_timeseries = {}
    backTestList = sorted(backTestList, key=lambda x: x.last_seen_period)
    for last_seen_period, forecast_list in itertools.groupby(backTestList, key=lambda x: x.last_seen_period):
        all_splitpoint_timeseries[last_seen_period] = convert_single_splitpoint_to_multi_location_forecast(
            forecast_list
        )
    return all_splitpoint_timeseries


def convert_single_splitpoint_to_multi_location_forecast(backTestList: List[BackTestForecast]) -> MultiLocationForecast:
    location_forecasts: Dict[str, List[Samples]] = defaultdict(list)
    for forecast in backTestList:
        location_key = str(forecast.org_unit)  # Or use forecast.backtest.location if available

        sample = Samples(
            time_period=str(forecast.period),  # Convert PeriodID to str
            disease_case_samples=forecast.values,
        )
        location_forecasts[location_key].append(sample)

    # Sort each list of Samples by time_period before wrapping in Forecast
    timeseries = {
        location: Forecast(predictions=sorted(samples, key=lambda s: s.time_period))
        for location, samples in location_forecasts.items()
    }

    return MultiLocationForecast(timeseries=timeseries)


def convert_to_multi_location_timeseries(obs: List[ObservationBase]) -> MultiLocationDiseaseTimeSeries:
    grouped: defaultdict[str, List[DiseaseObservation]] = defaultdict(list)

    for ob in obs:
        if ob.feature_name == "disease_cases" and ob.value is not None:
            disease_obs = DiseaseObservation(
                time_period=str(ob.period),  # Ensure PeriodID is string-convertible
                disease_cases=int(ob.value),  # Round or cast as needed
            )
            grouped[ob.org_unit].append(disease_obs)

    multi_ts = MultiLocationDiseaseTimeSeries()
    for location, observations in grouped.items():
        # Optionally sort by time_period
        observations.sort(key=lambda x: x.time_period)
        multi_ts[location] = DiseaseTimeSeries(observations=observations)

    return multi_ts


def mean(samples):
    return sum(samples) / len(samples)


class MAEonMeanPredictions(Evaluator):
    # def evaluate(self, true_values, samples):
    def evaluate(
        self, all_truths: MultiLocationDiseaseTimeSeries, all_forecasts: MultiLocationForecast
    ) -> MultiLocationErrorTimeSeries:
        evaluation_result = MultiLocationErrorTimeSeries(timeseries_dict={})
        for location in all_truths.locations():
            truth_series = all_truths[location]
            forecast_series = all_forecasts.timeseries[location]
            assert len(truth_series.observations) == len(forecast_series.predictions), (
                f"{len(truth_series.observations)} != {len(forecast_series.predictions)}"
            )
            truth_and_forecast_series = zip(truth_series.observations, forecast_series.predictions)
            error = 0
            for truth, prediction in truth_and_forecast_series:
                assert truth.time_period == prediction.time_period, (truth.time_period, prediction.time_period)
                predicted_mean = mean(prediction.disease_case_samples)
                error += abs(truth.disease_cases - predicted_mean)

            mean_absolute_error = error / len(truth_series.observations)
            evaluation_result[location] = ErrorTimeSeries(
                observations=[Error(time_period="Full_period", value=mean_absolute_error)]
            )
        return evaluation_result

    # def evaluate_and_give_one_aggregated_metric(...) -> float:
    #    res = self.evaluate()
    #    for location, ..
