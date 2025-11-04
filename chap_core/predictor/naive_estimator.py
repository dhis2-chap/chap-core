import numpy as np
import json
import dataclasses

from chap_core import get_temp_dir
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.datatypes import Samples


@dataclasses.dataclass
class NaivePredictor:
    mean_dict: dict

    def predict(self, historic_data: DataSet, future_data: DataSet, num_samples: int = 100) -> DataSet:
        # write future_data to from a tmp csv file (to mimic what is happening in chap)
        test_csv_path = get_temp_dir() / "test.csv"
        test_csv_path.parent.mkdir(parents=True, exist_ok=True)
        future_data.to_csv(str(test_csv_path))
        future_data = DataSet.from_csv(str(test_csv_path))

        samples = DataSet(
            {
                location: Samples(
                    future_data[location].time_period,
                    np.random.poisson(
                        self.mean_dict[location] if not np.isnan(self.mean_dict[location]) else 0,
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
