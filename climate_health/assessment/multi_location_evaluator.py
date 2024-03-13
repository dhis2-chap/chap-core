from typing import List

from climate_health.dataset import IsSpatioTemporalDataSet
from climate_health.datatypes import HealthData, ClimateHealthTimeSeries, ResultType


class MultiLocationEvaluator:
    def __init__(self, model_names: List[str], truth: IsSpatioTemporalDataSet[HealthData]):
        self.model_names = model_names
        self.truth = truth
        self.predictions = {model_name: [] for model_name in model_names}

    def add_predictions(self, model_name: str, predictions: IsSpatioTemporalDataSet[HealthData]):
        return NotImplemented

    def get_results(self)-> dict[str, ResultType]:
        return NotImplemented