import numpy as np

from chap_core.database.model_templates_and_config_tables import ModelTemplateRole
from chap_core.datatypes import Samples
from chap_core.models.builtin.registry import BuiltinModel, BuiltinModelSpec, builtin_model
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import Month, TimePeriod

N_SAMPLES = 100
SEED = 0


def _period_of_year(period: TimePeriod) -> int:
    return int(period.month if isinstance(period, Month) else period.week)  # type: ignore[attr-defined]


@builtin_model()
class SeasonalMedian(BuiltinModel):
    """Forecasts a period with all observations of the same location and period of year in the training data.

    Periods of year have different numbers of observations, so they are resampled
    with replacement to ``N_SAMPLES`` samples.
    """

    spec = BuiltinModelSpec(
        name="seasonal_median",
        display_name="Seasonal median",
        description=(
            "Baseline that uses all observed values for the same location and month or week of the year "
            "as forecast samples, without smoothing or model fitting."
        ),
        version="1",
        role=ModelTemplateRole.baseline,
    )

    def __init__(self):
        self._observations: dict[str, dict[int, np.ndarray]] = {}

    def train(self, data: DataSet) -> None:
        for location, location_data in data.items():
            cases = np.asarray(location_data.disease_cases, dtype=float)
            periods = np.array([_period_of_year(period) for period in location_data.time_period])
            observed = np.isfinite(cases)
            self._observations[location] = {
                int(period): cases[observed & (periods == period)] for period in np.unique(periods[observed])
            }

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        rng = np.random.default_rng(SEED)
        forecasts = {}
        for location, location_data in future_data.items():
            if location not in self._observations:
                raise ValueError(f"Location {location!r} was not in the training data")
            samples = []
            for period in location_data.time_period:
                values = self._observations[location].get(_period_of_year(period))
                if values is None:
                    raise ValueError(
                        f"No observed disease_cases for location {location!r} in the same period of year as {period}"
                    )
                samples.append(rng.choice(values, size=N_SAMPLES))
            forecasts[location] = Samples(location_data.time_period, np.array(samples))  # type: ignore[call-arg]
        return DataSet(forecasts)
