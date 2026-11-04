from dataclasses import dataclass

import numpy as np

from chap_core.database.model_templates_and_config_tables import ModelTemplateRole
from chap_core.datatypes import Samples
from chap_core.models.builtin.empirical import quantile_samples
from chap_core.models.builtin.registry import BaselineNotApplicable, BuiltinModel, BuiltinModelSpec, builtin_model
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


@dataclass(frozen=True)
class GlobalMedianSamples:
    """Quantile samples of all observed disease_cases of each location in the training data."""

    by_location: dict[str, np.ndarray]

    @classmethod
    def from_data(cls, data: DataSet) -> "GlobalMedianSamples":
        by_location = {}
        for location, location_data in data.items():
            cases = np.asarray(location_data.disease_cases, dtype=float)
            observed = cases[np.isfinite(cases)]
            if len(observed) > 0:
                by_location[location] = quantile_samples(observed)
        return cls(by_location)

    def for_location(self, location: str) -> np.ndarray:
        """The samples of the location, or the median of the samples of all other locations when it has none.

        The median is taken per quantile level, so the samples stay sorted and their median is the
        median of the other locations' medians.
        """
        if location in self.by_location:
            return self.by_location[location]
        if not self.by_location:
            raise BaselineNotApplicable("No observed disease_cases for any location in the training data")
        return np.asarray(np.median(np.stack(list(self.by_location.values())), axis=0))


@builtin_model()
class GlobalMedian(BuiltinModel):
    """Forecasts every period with all observations of the same location in the training data.

    The samples are quantiles of those observations, so their median is the median of
    the observations. A location without observations gets the median of the other
    locations' samples.
    """

    spec = BuiltinModelSpec(
        name="global_median",
        display_name="Global median",
        description=(
            "Baseline that uses all observed values for the same location as forecast samples for every "
            "period, without smoothing or model fitting."
        ),
        version="2",
        role=ModelTemplateRole.baseline,
    )

    def train(self, data: DataSet) -> None:
        self._samples = GlobalMedianSamples.from_data(data)

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        forecasts = {}
        for location, location_data in future_data.items():
            samples = np.tile(self._samples.for_location(location), (len(location_data.time_period), 1))
            forecasts[location] = Samples(location_data.time_period, samples)  # type: ignore[call-arg]
        return DataSet(forecasts)
