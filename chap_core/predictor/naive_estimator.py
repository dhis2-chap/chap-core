
import numpy as np
import json
import dataclasses
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.datatypes import Samples


@dataclasses.dataclass
class NaivePredictor:
    mean_dict: dict

    def predict(self, historic_data: DataSet, future_data: DataSet, num_samples: int = 100) -> DataSet:
        samples = DataSet(
            {
                location: Samples(
                    future_data[location].time_period,
                    np.random.poisson(
                        self.mean_dict[location],
                        len(future_data[location]) * num_samples,
                    ).reshape(-1, num_samples),
                )
                for location in future_data.keys()
            }
        )
        return samples

    def save(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self.mean_dict, f)

    @classmethod
    def load(cls, filename: str):
        with open(filename, "r") as f:
            mean_dict = json.load(f)
        return cls(mean_dict)


class NaiveEstimator:
    def train(self, data: DataSet) -> NaivePredictor:
        mean_dict = {location: np.nanmean(data[location].disease_cases) for location in data.keys()}
        return NaivePredictor(mean_dict)
