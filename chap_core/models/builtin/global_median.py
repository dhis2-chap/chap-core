from dataclasses import dataclass

import numpy as np

from chap_core.database.model_templates_and_config_tables import ModelTemplateRole
from chap_core.datatypes import Samples
from chap_core.models.builtin.empirical import even_samples, sample_count
from chap_core.models.builtin.registry import BaselineNotApplicable, BuiltinModel, BuiltinModelSpec, builtin_model
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


@dataclass(frozen=True)
class GlobalObservations:
    """All observed disease_cases of each location in the training data."""

    by_location: dict[str, np.ndarray]

    @classmethod
    def from_data(cls, data: DataSet) -> "GlobalObservations":
        by_location = {}
        for location, location_data in data.items():
            cases = np.asarray(location_data.disease_cases, dtype=float)
            observed = cases[np.isfinite(cases)]
            if len(observed) > 0:
                by_location[location] = observed
        return cls(by_location)

    @property
    def largest_group(self) -> int:
        """The number of observations of the location with the most."""
        return max((len(observed) for observed in self.by_location.values()), default=1)

    def for_location(self, location: str) -> np.ndarray:
        """The observations of the location, or those of all locations when it has none."""
        if location in self.by_location:
            return self.by_location[location]
        if not self.by_location:
            raise BaselineNotApplicable("No observed disease_cases for any location in the training data")
        return np.concatenate(list(self.by_location.values()))


@builtin_model()
class GlobalMedian(BuiltinModel):
    """Forecasts every period with all observations of the same location in the training data.

    The samples are the observations, each repeated about equally often, with their exact
    median as the middle sample (see ``even_samples`` and ``sample_count``). A location
    without observations gets the observations of all other locations.
    """

    spec = BuiltinModelSpec(
        name="global_median",
        display_name="Global median",
        description=(
            "Baseline that uses all observed values for the same location as forecast samples for every "
            "period, without smoothing or model fitting."
        ),
        version="3",
        role=ModelTemplateRole.baseline,
    )

    def train(self, data: DataSet) -> None:
        self._observations = GlobalObservations.from_data(data)
        self._n_samples = sample_count(self._observations.largest_group)

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        forecasts = {}
        for location, location_data in future_data.items():
            samples = np.tile(
                even_samples(self._observations.for_location(location), self._n_samples),
                (len(location_data.time_period), 1),
            )
            forecasts[location] = Samples(location_data.time_period, samples)  # type: ignore[call-arg]
        return DataSet(forecasts)
