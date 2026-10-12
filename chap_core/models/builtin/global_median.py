import numpy as np

from chap_core.database.model_templates_and_config_tables import ModelTemplateRole
from chap_core.datatypes import Samples
from chap_core.models.builtin.registry import BuiltinModel, BuiltinModelSpec, builtin_model
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

N_SAMPLES = 100
SEED = 0


@builtin_model()
class GlobalMedian(BuiltinModel):
    """Forecasts every period with all observations of the same location in the training data.

    Locations have different numbers of observations, so they are resampled with
    replacement to ``N_SAMPLES`` samples.
    """

    spec = BuiltinModelSpec(
        name="global_median",
        display_name="Global median",
        description=(
            "Baseline that uses all observed values for the same location as forecast samples for every "
            "period, without smoothing or model fitting."
        ),
        version="1",
        role=ModelTemplateRole.baseline,
    )

    def __init__(self):
        self._observations: dict[str, np.ndarray] = {}

    def train(self, data: DataSet) -> None:
        for location, location_data in data.items():
            cases = np.asarray(location_data.disease_cases, dtype=float)
            self._observations[location] = cases[np.isfinite(cases)]

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        rng = np.random.default_rng(SEED)
        forecasts = {}
        for location, location_data in future_data.items():
            values = self._observations.get(location)
            if values is None or len(values) == 0:
                raise ValueError(f"No observed disease_cases for location {location!r} in the training data")
            samples = rng.choice(values, size=(len(location_data.time_period), N_SAMPLES))
            forecasts[location] = Samples(location_data.time_period, samples)  # type: ignore[call-arg]
        return DataSet(forecasts)
