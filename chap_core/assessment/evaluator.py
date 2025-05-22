from abc import ABC, abstractmethod

from chap_core.assessment.representations import (
    MultiLocationDiseaseTimeSeries,
    MultiLocationForecast,
    MultiLocationErrorTimeSeries,
    ErrorTimeSeries,
    Error,
)


class Evaluator(ABC):
    @abstractmethod
    def evaluate(
        self, all_truths: MultiLocationDiseaseTimeSeries, all_forecasts: MultiLocationForecast
    ) -> MultiLocationErrorTimeSeries:
        pass

    def get_name(self) -> str:
        return self.__class__.__name__


class ComponentBasedEvaluator(Evaluator):
    def __init__(self, name, errorFunc, timeAggregationFunc, regionAggregationFunc):
        self._name = name
        self._errorFunc = errorFunc
        self._timeAggregationFunc = timeAggregationFunc
        self._regionAggregationFunc = regionAggregationFunc

    def get_name(self):
        return self._name

    def evaluate(
        self, all_truths: MultiLocationDiseaseTimeSeries, all_forecasts: MultiLocationForecast
    ) -> MultiLocationErrorTimeSeries:
        evaluation_result = MultiLocationErrorTimeSeries(timeseries_dict={})
        for location in all_truths.locations():
            current_error_series = ErrorTimeSeries(observations=[])
            forecast_series = all_forecasts.timeseries[location]
            assert len(all_truths[location].observations) == len(forecast_series.predictions)
            truth_and_forecast_series = zip(all_truths[location].observations, forecast_series.predictions)
            errors = []
            for truth, prediction in truth_and_forecast_series:
                assert truth.time_period == prediction.time_period
                errors.append(self._errorFunc(truth.disease_cases, prediction.disease_case_samples))
                if self._timeAggregationFunc is None:
                    current_error_series.observations.append(Error(time_period=truth.time_period, value=errors[-1]))
            if self._timeAggregationFunc is not None:
                current_error_series.observations.append(
                    Error(time_period="Full_period", value=self._timeAggregationFunc(errors))
                )
            evaluation_result[location] = current_error_series

        if self._regionAggregationFunc is not None:
            final_evaluation_result = MultiLocationErrorTimeSeries(
                timeseries_dict={"Full_region": ErrorTimeSeries(observations=[])}
            )
            for locationvalues in evaluation_result.locationvalues_per_timepoint():
                aggregated_error = self._regionAggregationFunc([error.value for error in locationvalues.values()])
                final_evaluation_result["Full_region"].observations.append(
                    Error(time_period="Full_period", value=aggregated_error)
                )
        else:
            final_evaluation_result = evaluation_result

        return final_evaluation_result
