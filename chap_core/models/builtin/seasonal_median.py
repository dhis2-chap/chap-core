from dataclasses import dataclass

import numpy as np

from chap_core.database.model_templates_and_config_tables import ModelTemplateRole
from chap_core.datatypes import Samples
from chap_core.models.builtin.empirical import period_of_year, quantile_samples
from chap_core.models.builtin.global_median import GlobalMedianSamples
from chap_core.models.builtin.registry import BuiltinModel, BuiltinModelSpec, builtin_model
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


@dataclass(frozen=True)
class SeasonalMedianSamples:
    """Quantile samples of the observed disease_cases of each location and period of year in the training data."""

    by_location: dict[str, dict[int, np.ndarray]]

    @classmethod
    def from_data(cls, data: DataSet) -> "SeasonalMedianSamples":
        by_location = {}
        for location, location_data in data.items():
            cases = np.asarray(location_data.disease_cases, dtype=float)
            seasons = period_of_year(location_data.time_period)
            observed = np.isfinite(cases)
            by_location[location] = {
                int(season): quantile_samples(cases[observed & (seasons == season)])
                for season in np.unique(seasons[observed])
            }
        return cls(by_location)

    def get(self, location: str, season: int) -> np.ndarray | None:
        return self.by_location.get(location, {}).get(season)

    def is_observed(self, location: str) -> bool:
        return bool(self.by_location.get(location))

    def across_locations(self, season: int) -> np.ndarray | None:
        """The median per quantile level of all locations' samples for the period of year, None when none has any."""
        samples = [by_season[season] for by_season in self.by_location.values() if season in by_season]
        if not samples:
            return None
        return np.asarray(np.median(np.stack(samples), axis=0))


@builtin_model()
class SeasonalMedian(BuiltinModel):
    """Forecasts each period with all observations of the same location and period of year.

    The samples are quantiles of those observations, so their median is the median of
    the observations. Week 53 is pooled with week 52. A period of year that was never
    observed for the location is forecast with the global median. A location without
    observations gets the median of the other locations' samples for the same period of
    year, and the global median when no location observed it.
    """

    spec = BuiltinModelSpec(
        name="seasonal_median",
        display_name="Seasonal median",
        description=(
            "Baseline that uses all observed values for the same location and month or week of the year "
            "as forecast samples, without smoothing or model fitting."
        ),
        version="2",
        role=ModelTemplateRole.baseline,
    )

    def train(self, data: DataSet) -> None:
        self._samples = SeasonalMedianSamples.from_data(data)
        self._global_median = GlobalMedianSamples.from_data(data)

    def _seasonal_samples(self, location: str, season: int) -> np.ndarray:
        samples = self._samples.get(location, season)
        if samples is None and not self._samples.is_observed(location):
            samples = self._samples.across_locations(season)
        if samples is None:
            return self._global_median.for_location(location)
        return samples

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        forecasts = {}
        for location, location_data in future_data.items():
            rows = [self._seasonal_samples(location, season) for season in period_of_year(location_data.time_period)]
            forecasts[location] = Samples(location_data.time_period, np.array(rows))  # type: ignore[call-arg]
        return DataSet(forecasts)
