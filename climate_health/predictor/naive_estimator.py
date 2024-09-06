import numpy as np
import dataclasses
from climate_health.spatio_temporal_data.temporal_dataclass import DataSet
from climate_health.datatypes import Samples

@dataclasses.dataclass
class NaivePredictor:
    mean_dict: dict

    def predict(self, historic_data: DataSet, future_data: DataSet, num_samples: int=100) -> DataSet:
        samples = DataSet({location: Samples(future_data[location].time_period,
                                             np.random.poisson(self.mean_dict[location], len(future_data[location])*num_samples).reshape(-1, num_samples))
                           for location in future_data.keys()})
        return samples

class NaiveEstimator:

    def train(self, data: DataSet) -> NaivePredictor:
        mean_dict = {location: np.nanmean(data[location].disease_cases)
                     for location in data.keys()}
        return NaivePredictor(mean_dict)
