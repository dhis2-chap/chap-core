from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from chap_core.database.model_templates_and_config_tables import ModelTemplateRole
from chap_core.datatypes import Samples
from chap_core.models.builtin.empirical import even_samples, period_of_year, periods_per_year, sample_count
from chap_core.models.builtin.registry import BaselineNotApplicable, BuiltinModel, BuiltinModelSpec, builtin_model
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


@dataclass(frozen=True)
class SeasonalShifts:
    """Ratios ``(y_t + 1) / (y_{t-h} + 1)`` by horizon ``h`` and the period of year of ``t``."""

    by_horizon: dict[int, dict[int, np.ndarray]]

    @classmethod
    def from_cases(cls, cases: np.ndarray, seasons: np.ndarray, max_horizon: int) -> "SeasonalShifts":
        by_horizon = {}
        for horizon in range(1, min(max_horizon, len(cases) - 1) + 1):
            ratios = (cases[horizon:] + 1) / (cases[:-horizon] + 1)
            target_seasons = seasons[horizon:]
            finite = np.isfinite(ratios)
            by_horizon[horizon] = {
                int(season): ratios[finite & (target_seasons == season)] for season in np.unique(target_seasons[finite])
            }
        return cls(by_horizon)

    @classmethod
    def pooled(cls, shifts: Iterable["SeasonalShifts"]) -> "SeasonalShifts":
        """The ratios of all the given shifts together."""
        collected: dict[int, dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
        for location_shifts in shifts:
            for horizon, by_season in location_shifts.by_horizon.items():
                for season, ratios in by_season.items():
                    collected[horizon][season].append(ratios)
        return cls(
            {
                horizon: {season: np.concatenate(ratios) for season, ratios in by_season.items()}
                for horizon, by_season in collected.items()
            }
        )

    @property
    def n_ratios(self) -> int:
        """The number of ratios of the horizon and period of year with the most, 0 without any."""
        return max((len(ratios) for by_season in self.by_horizon.values() for ratios in by_season.values()), default=0)

    def ratios(self, season: int, horizon: int) -> np.ndarray | None:
        """The ratios ending in the period of year, else in the nearest ones earlier and later in the year.

        Without ratios both earlier and later in the year, the ratios of every period of year
        are used. None when there are no ratios for the horizon.
        """
        by_season = self.by_horizon.get(horizon, {})
        if not by_season:
            return None
        if season in by_season:
            return by_season[season]
        earlier = [other for other in by_season if other < season]
        later = [other for other in by_season if other > season]
        if earlier and later:
            return np.concatenate([by_season[max(earlier)], by_season[min(later)]])
        return np.concatenate(list(by_season.values()))


@builtin_model()
class SeasonalShiftPersistence(BuiltinModel):
    """Forecasts the last observed value, scaled by how cases changed in earlier years.

    For a target ``h`` periods after the last observed value ``y``, the samples are
    ``(y + 1) * r - 1`` for the ratios ``r = (y_t + 1) / (y_{t-h} + 1)`` in the training
    data of the same location, over every ``t`` with the target's period of year. The
    ratios are repeated about equally often around their exact median as the middle sample
    (see ``even_samples`` and ``sample_count``), so the median forecast is ``y`` scaled by the
    median ratio, and the spread widens with the horizon. Adding one keeps zero counts usable. Week 53 is
    pooled with week 52.

    Training keeps the ratios for horizons up to one year. Missing ratios fall back to
    those of the nearest periods of year (see ``SeasonalShifts.ratios``), then to those of
    all locations. A horizon without ratios, such as one longer than the training data
    or than a year, uses the longest shorter horizon that has them. A single ratio gives
    no spread, so more ratios found further down this order are preferred. Without any
    ratio the last value is forecast unchanged. A location without an observed value in
    the historic data has no value to persist, so it is not applicable.
    """

    spec = BuiltinModelSpec(
        name="seasonal_shift_persistence",
        display_name="Seasonal shift persistence",
        description=(
            "Baseline that forecasts the last observed value for the same location, scaled by how cases "
            "changed over the same horizon and time of year in the training data."
        ),
        version="2",
        role=ModelTemplateRole.baseline,
    )

    def train(self, data: DataSet) -> None:
        max_horizon = periods_per_year(data.period_range)
        self._shifts = {
            location: SeasonalShifts.from_cases(
                np.asarray(location_data.disease_cases, dtype=float),
                period_of_year(location_data.time_period),
                max_horizon,
            )
            for location, location_data in data.items()
        }
        self._pooled_shifts = SeasonalShifts.pooled(self._shifts.values())
        self._n_samples = sample_count(max((shifts.n_ratios for shifts in self._shifts.values()), default=0))

    def _ratios(self, location: str, season: int, horizon: int) -> np.ndarray:
        """The ratios of the location, else of all locations, at the horizon or else the longest shorter one.

        A single ratio gives no spread, so the first candidate with more than one is used, else
        the first single ratio, else 1.
        """
        candidates = [shifts for shifts in (self._shifts.get(location), self._pooled_shifts) if shifts is not None]
        single = None
        for shorter_horizon in range(horizon, 0, -1):
            for shifts in candidates:
                ratios = shifts.ratios(season, shorter_horizon)
                if ratios is not None and len(ratios) > 1:
                    return ratios
                if single is None:
                    single = ratios
        return np.ones(1) if single is None else single

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        forecasts = {}
        for location, location_data in future_data.items():
            cases = np.asarray(historic_data[location].disease_cases, dtype=float)
            observed = np.flatnonzero(np.isfinite(cases))
            if len(observed) == 0:
                raise BaselineNotApplicable(f"No observed disease_cases for location {location!r} in the historic data")
            last_value = cases[observed[-1]]
            periods_since_last = len(cases) - observed[-1]
            rows = [
                np.maximum(
                    (last_value + 1) * even_samples(self._ratios(location, season, horizon), self._n_samples) - 1, 0
                )
                for horizon, season in enumerate(period_of_year(location_data.time_period), start=periods_since_last)
            ]
            forecasts[location] = Samples(location_data.time_period, np.array(rows))  # type: ignore[call-arg]
        return DataSet(forecasts)
