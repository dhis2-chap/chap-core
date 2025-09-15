import numpy as np


import abc

import pydantic


class Metric(abc.ABC):
    @abc.abstractmethod
    def compute(self, truth: float, predictions: np.ndarray) -> float:
        ...

    def aggregate(self, values: list[float]) -> float:
        return float(np.mean(values))

    def finalize(self, value: float) -> float:
        return value



class CRPS(Metric):
    def compute(self, truth, predictions) -> float:
        term1 = np.mean(np.abs(predictions - truth))
        term2 = 0.5 * np.mean(np.abs(predictions[:, None] - predictions[None, :]))
        return float(term1 - term2)


class AggregationInfo(pydantic.BaseModel):
    '''
    This class describes whether a metric is aggregated over temporal or spatial dimensions before being put in the database.
    '''
    temporal: bool = False
    spatial: bool = False
    samples: bool = True


def get_aggreagtion_level(metric_id: str):
    return AggregationInfo(temporal=False, spatial=False)


class MetricRegistry:
    metrics = ['crps_mean',
               'crps_norm_mean',
               'ratio_within_10th_90th',
               'ratio_within_90th']
    seeded_aggregations = {name: np.mean for name in metrics}

    def get_aggregation_function(self, metric_id: str):
        # Placeholder for actual implementation
        return self.seeded_aggregations.get(metric_id)