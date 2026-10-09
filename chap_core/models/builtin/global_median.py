import numpy as np

from chap_core.database.model_templates_and_config_tables import ModelTemplateRole
from chap_core.datatypes import Samples
from chap_core.models.builtin.empirical import quantile_samples
from chap_core.models.builtin.registry import BuiltinModel, BuiltinModelSpec, builtin_model
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


@builtin_model()
class GlobalMedian(BuiltinModel):
    """Forecasts every period with all observations of the same location in the training data.

    The samples are quantiles of those observations, so their median is the median of
    the observations.
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
        self._samples: dict[str, np.ndarray] = {}

    def train(self, data: DataSet) -> None:
        for location, location_data in data.items():
            cases = np.asarray(location_data.disease_cases, dtype=float)
            observed = cases[np.isfinite(cases)]
            if len(observed) > 0:
                self._samples[location] = quantile_samples(observed)

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        forecasts = {}
        for location, location_data in future_data.items():
            if location not in self._samples:
                raise ValueError(f"No observed disease_cases for location {location!r} in the training data")
            samples = np.tile(self._samples[location], (len(location_data.time_period), 1))
            forecasts[location] = Samples(location_data.time_period, samples)  # type: ignore[call-arg]
        return DataSet(forecasts)
