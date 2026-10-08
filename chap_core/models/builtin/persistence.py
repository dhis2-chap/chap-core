import numpy as np

from chap_core.database.model_templates_and_config_tables import ModelTemplateRole
from chap_core.datatypes import Samples
from chap_core.models.builtin.empirical import period_of_year, quantile_samples
from chap_core.models.builtin.registry import BuiltinModel, BuiltinModelSpec, builtin_model
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


@builtin_model()
class Persistence(BuiltinModel):
    """Forecasts the last observed value, scaled by how cases changed in earlier years.

    For a target ``h`` periods after the last observed value ``y``, the samples are
    ``(y + 1) * r - 1`` for the ratios ``r = (y_t + 1) / (y_{t-h} + 1)`` in the training
    data of the same location, over every ``t`` with the target's period of year. The
    ratios are quantile samples, so the median is ``y`` scaled by the median ratio, and
    the spread widens with the horizon. Adding one keeps zero counts usable. Week 53 is
    pooled with week 52.
    """

    spec = BuiltinModelSpec(
        name="persistence",
        display_name="Persistence",
        description=(
            "Baseline that forecasts the last observed value for the same location, scaled by how cases "
            "changed over the same horizon and time of year in the training data."
        ),
        version="1",
        role=ModelTemplateRole.baseline,
    )

    def __init__(self):
        self._history: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def train(self, data: DataSet) -> None:
        for location, location_data in data.items():
            cases = np.asarray(location_data.disease_cases, dtype=float)
            self._history[location] = (cases, period_of_year(location_data.time_period))

    def _ratios(self, location: str, season: int, horizon: int) -> np.ndarray:
        if location not in self._history:
            raise ValueError(f"No training data for location {location!r}")
        cases, seasons = self._history[location]
        ratios = (cases[horizon:] + 1) / (cases[:-horizon] + 1)
        ratios = ratios[(seasons[horizon:] == season) & np.isfinite(ratios)]
        if len(ratios) == 0:
            raise ValueError(
                f"No pair of observations {horizon} periods apart ending in period of year {season} for location "
                f"{location!r} in the training data. Persistence needs one to estimate the spread."
            )
        return np.asarray(ratios)

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        forecasts = {}
        for location, location_data in future_data.items():
            cases = np.asarray(historic_data[location].disease_cases, dtype=float)
            observed = np.flatnonzero(np.isfinite(cases))
            if len(observed) == 0:
                raise ValueError(f"No observed disease_cases for location {location!r} in the historic data")
            last_value = cases[observed[-1]]
            periods_since_last = len(cases) - observed[-1]
            rows = [
                np.maximum((last_value + 1) * quantile_samples(self._ratios(location, season, horizon)) - 1, 0)
                for horizon, season in enumerate(period_of_year(location_data.time_period), start=periods_since_last)
            ]
            forecasts[location] = Samples(location_data.time_period, np.array(rows))  # type: ignore[call-arg]
        return DataSet(forecasts)
