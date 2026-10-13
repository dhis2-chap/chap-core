import numpy as np

from chap_core.database.model_templates_and_config_tables import ModelTemplateRole
from chap_core.datatypes import Samples
from chap_core.models.builtin.empirical import period_of_year, quantile_samples
from chap_core.models.builtin.registry import BuiltinModel, BuiltinModelSpec, builtin_model
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


@builtin_model()
class SeasonalMedian(BuiltinModel):
    """Forecasts each period with all observations of the same location and period of year.

    The samples are quantiles of those observations, so their median is the median of
    the observations. Week 53 is pooled with week 52. Every period of year that is
    forecast must be observed at least once in the training data.
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
        self._samples: dict[str, dict[int, np.ndarray]] = {}

    def train(self, data: DataSet) -> None:
        for location, location_data in data.items():
            cases = np.asarray(location_data.disease_cases, dtype=float)
            seasons = period_of_year(location_data.time_period)
            observed = np.isfinite(cases)
            self._samples[location] = {
                int(season): quantile_samples(cases[observed & (seasons == season)])
                for season in np.unique(seasons[observed])
            }

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        forecasts = {}
        for location, location_data in future_data.items():
            by_season = self._samples.get(location, {})
            rows = []
            for season in period_of_year(location_data.time_period):
                if season not in by_season:
                    raise ValueError(
                        f"No observed disease_cases for location {location!r} in period of year {season} in the "
                        "training data. The seasonal median needs every forecast period of the year observed."
                    )
                rows.append(by_season[season])
            forecasts[location] = Samples(location_data.time_period, np.array(rows))  # type: ignore[call-arg]
        return DataSet(forecasts)
