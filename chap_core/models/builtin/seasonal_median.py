from dataclasses import dataclass

import numpy as np

from chap_core.database.model_templates_and_config_tables import ModelTemplateRole
from chap_core.datatypes import Samples
from chap_core.models.builtin.empirical import even_samples, period_of_year
from chap_core.models.builtin.global_median import GlobalObservations
from chap_core.models.builtin.registry import BuiltinModel, BuiltinModelSpec, builtin_model
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


@dataclass(frozen=True)
class SeasonalObservations:
    """The observed disease_cases of each location and period of year in the training data."""

    by_location: dict[str, dict[int, np.ndarray]]

    @classmethod
    def from_data(cls, data: DataSet) -> "SeasonalObservations":
        by_location = {}
        for location, location_data in data.items():
            cases = np.asarray(location_data.disease_cases, dtype=float)
            seasons = period_of_year(location_data.time_period)
            observed = np.isfinite(cases)
            by_location[location] = {
                int(season): cases[observed & (seasons == season)] for season in np.unique(seasons[observed])
            }
        return cls(by_location)

    @property
    def n_samples(self) -> int:
        """The number of observations of the location and period of year with the most."""
        return max(
            (len(observed) for by_season in self.by_location.values() for observed in by_season.values()), default=1
        )

    def get(self, location: str, season: int) -> np.ndarray | None:
        return self.by_location.get(location, {}).get(season)

    def is_observed(self, location: str) -> bool:
        return bool(self.by_location.get(location))

    def across_locations(self, season: int) -> np.ndarray | None:
        """The observations of all locations for the period of year, None when none has any."""
        observed = [by_season[season] for by_season in self.by_location.values() if season in by_season]
        if not observed:
            return None
        return np.concatenate(observed)


@builtin_model()
class SeasonalMedian(BuiltinModel):
    """Forecasts each period with all observations of the same location and period of year.

    The samples are the observations themselves, for the location and period of year
    with the most observations, and the observations spread evenly over as many samples for
    the others (see ``even_samples``). Week 53 is pooled with week 52. A period of year that
    was never observed for the location uses all observations of the location, thinned
    evenly to the same number of samples. A location without observations uses the
    observations of all other locations for the same period of year, and the global median's
    observations when no location observed it.
    """

    spec = BuiltinModelSpec(
        name="seasonal_median",
        display_name="Seasonal median",
        description=(
            "Baseline that uses all observed values for the same location and month or week of the year "
            "as forecast samples, without smoothing or model fitting."
        ),
        version="3",
        role=ModelTemplateRole.baseline,
    )

    def train(self, data: DataSet) -> None:
        self._observations = SeasonalObservations.from_data(data)
        self._n_samples = self._observations.n_samples
        self._global_observations = GlobalObservations.from_data(data)

    def _seasonal_observations(self, location: str, season: int) -> np.ndarray:
        observed = self._observations.get(location, season)
        if observed is None and not self._observations.is_observed(location):
            observed = self._observations.across_locations(season)
        if observed is None:
            return self._global_observations.for_location(location)
        return observed

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        forecasts = {}
        for location, location_data in future_data.items():
            rows = [
                even_samples(self._seasonal_observations(location, season), self._n_samples)
                for season in period_of_year(location_data.time_period)
            ]
            forecasts[location] = Samples(location_data.time_period, np.array(rows))  # type: ignore[call-arg]
        return DataSet(forecasts)
