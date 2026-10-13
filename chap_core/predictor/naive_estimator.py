import dataclasses
import json

import numpy as np

from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB, ModelTemplateDB
from chap_core.datatypes import Samples
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


@dataclasses.dataclass
class NaivePredictor:
    mean_dict: dict

    def predict(self, historic_data: DataSet, future_data: DataSet, num_samples: int = 100) -> DataSet:
        samples = DataSet(
            {
                location: Samples(
                    future_data[location].time_period,
                    np.random.poisson(
                        self.mean_dict[location] if not np.isnan(self.mean_dict[location]) else 0,
                        len(future_data[location]) * num_samples,
                    ).reshape(-1, num_samples),
                )  # type: ignore[call-arg]
                for location in future_data.keys()  # noqa: SIM118
            }
        )
        return samples

    def save(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self.mean_dict, f)

    @classmethod
    def load(cls, filename: str):
        with open(filename) as f:
            mean_dict = json.load(f)
        return cls(mean_dict)


class NaiveEstimator:
    model_template_db = ModelTemplateDB(id=1, name="naive_model", version="1.0")
    configured_model_db = ConfiguredModelDB(
        id="naive_eval", name="naive_configured", model_template_id=1, model_template=model_template_db
    )

    def train(self, data: DataSet) -> NaivePredictor:
        mean_dict = {location: np.nanmean(data[location].disease_cases) for location in data.keys()}  # noqa: SIM118
        return NaivePredictor(mean_dict)
