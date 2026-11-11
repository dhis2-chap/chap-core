from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from chap_core.database.model_templates_and_config_tables import ModelTemplateRole
from chap_core.datatypes import Samples
from chap_core.models.builtin.empirical import even_samples, periods_per_year, sample_count
from chap_core.models.builtin.registry import BaselineNotApplicable, BuiltinModel, BuiltinModelSpec, builtin_model
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


@dataclass(frozen=True)
class LogDifferences:
    """Differences ``log(y_t + 1) - log(y_{t-h} + 1)`` by horizon ``h``."""

    by_horizon: dict[int, np.ndarray]

    @classmethod
    def from_cases(cls, cases: np.ndarray, max_horizon: int) -> "LogDifferences":
        log_cases = np.log(cases + 1)
        by_horizon = {}
        for horizon in range(1, min(max_horizon, len(cases) - 1) + 1):
            differences = log_cases[horizon:] - log_cases[:-horizon]
            differences = differences[np.isfinite(differences)]
            if len(differences) > 0:
                by_horizon[horizon] = differences
        return cls(by_horizon)

    @classmethod
    def pooled(cls, differences: Iterable["LogDifferences"]) -> "LogDifferences":
        """The differences of all the given ones together."""
        collected: dict[int, list[np.ndarray]] = defaultdict(list)
        for location_differences in differences:
            for horizon, values in location_differences.by_horizon.items():
                collected[horizon].append(values)
        return cls({horizon: np.concatenate(values) for horizon, values in collected.items()})

    @property
    def n_differences(self) -> int:
        """The number of differences of the horizon with the most, 0 without any."""
        return max((len(values) for values in self.by_horizon.values()), default=0)


@builtin_model()
class Persistence(BuiltinModel):
    """Forecasts the last observed value, with a spread from how cases changed over the same horizon.

    For a target ``h`` periods after the last observed value ``y``, the samples are
    ``(y + 1) * exp(d - median(d)) - 1`` for the differences
    ``d = log(y_t + 1) - log(y_{t-h} + 1)`` over every ``t`` in the training data of the
    same location, each repeated about equally often around their exact median as the middle
    sample (see ``even_samples`` and ``sample_count``). Centring the samples on that median
    keeps a trend in the training data from moving the forecast, and makes the middle sample
    exactly ``y``. The spread widens with the horizon.

    Training keeps the differences for horizons up to one year. Missing differences fall
    back to those of all locations. A horizon without differences, such as one longer than
    the training data or than a year, uses the longest shorter horizon that has them. A
    single difference gives no spread after centring, so at least two are needed. Without
    them the last value is forecast unchanged. A location
    without an observed value in the historic data has no value to persist, so it is not
    applicable.
    """

    spec = BuiltinModelSpec(
        name="persistence",
        display_name="Persistence",
        description=(
            "Baseline that forecasts the last observed value for the same location, with a spread from "
            "how cases changed over the same horizon in the training data."
        ),
        # Version 1 of this name was the seasonal shift model, now seasonal_shift_persistence.
        version="3",
        role=ModelTemplateRole.baseline,
    )

    def train(self, data: DataSet) -> None:
        max_horizon = periods_per_year(data.period_range)
        self._differences = {
            location: LogDifferences.from_cases(np.asarray(location_data.disease_cases, dtype=float), max_horizon)
            for location, location_data in data.items()
        }
        self._pooled_differences = LogDifferences.pooled(self._differences.values())
        self._n_samples = sample_count(
            max((differences.n_differences for differences in self._differences.values()), default=0)
        )

    def _centred_differences(self, location: str, horizon: int) -> np.ndarray:
        """Samples of the differences of the location, else of all locations, at the horizon.

        Without differences at the horizon, the longest shorter horizon with them is used.

        The samples are centred on their median, so a single difference gives no spread and
        needs at least two. Without any, 0.
        """
        candidates = [
            differences
            for differences in (self._differences.get(location), self._pooled_differences)
            if differences is not None
        ]
        for shorter_horizon in range(horizon, 0, -1):
            for differences in candidates:
                values = differences.by_horizon.get(shorter_horizon)
                if values is not None and len(values) > 1:
                    samples = even_samples(values, self._n_samples)
                    return np.asarray(samples - np.median(samples))
        return np.zeros(self._n_samples)

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
                np.maximum((last_value + 1) * np.exp(self._centred_differences(location, horizon)) - 1, 0)
                for horizon in range(periods_since_last, periods_since_last + len(location_data.time_period))
            ]
            forecasts[location] = Samples(location_data.time_period, np.array(rows))  # type: ignore[call-arg]
        return DataSet(forecasts)
