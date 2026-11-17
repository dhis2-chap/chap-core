from __future__ import annotations

import functools
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera import Check
from pandera.pandas import DataFrameModel

from chap_core.time_period import TimePeriod

if TYPE_CHECKING:
    from collections.abc import Mapping

    from chap_core.database.dataset_tables import ObservationBase
    from chap_core.database.tables import BacktestForecast


class FlatData(DataFrameModel):
    """
    Base class for data points that include location and time_period.
    """

    location: pa.typing.Series[str]
    time_period: pa.typing.Series[str]


class FlatDataWithHorizon(FlatData):
    horizon_distance: pa.typing.Series[int]


class FlatObserved(FlatData):
    """
    Observed disease cases, with the population of each location and period when the
    dataset has one (needed by incidence-based metrics).
    """

    disease_cases: pa.typing.Series[float] = pa.Field(nullable=True)  # float to also allow nan
    population: pa.typing.Series[float] | None = pa.Field(nullable=True)


class FlatForecasts(FlatDataWithHorizon):
    """
    Forecasted disease cases. Note that cases are in forecast field, and that samples is used
    so we can represent multiple samples per location/time_period/horizon_distance in the dataframe.
    """

    sample: pa.typing.Series[int]  # index of sample
    forecast: pa.typing.Series[float]  # actual forecast value


class FlatMetric(FlatDataWithHorizon):
    metric: pa.typing.Series[float] = pa.Field(nullable=True)


@functools.lru_cache(maxsize=4096)
def horizon_diff(period: str, period2: str) -> int:
    """Calculate the 1-based horizon distance between two time periods.

    Returns the number of time steps from period2 to period, counting from 1.
    E.g. if period2 is the last seen period and period is the next one, returns 1.
    """
    tp = TimePeriod.parse(period)
    tp2 = TimePeriod.parse(period2)
    return int((tp - tp2) // tp.time_delta) + 1


def max_horizon_distance(backtest_forecasts: list[BacktestForecast]) -> int | None:
    """Largest 1-based horizon distance across a backtest's forecasts, or None if empty."""
    if not backtest_forecasts:
        return None
    return max(horizon_diff(str(fc.period), str(fc.last_seen_period)) for fc in backtest_forecasts)


def _create_df(forecast: BacktestForecast, horizon_distance: int):
    df = pd.DataFrame(
        {
            "location": str(forecast.org_unit),
            "time_period": str(forecast.period),
            "horizon_distance": horizon_distance,
            "sample": np.arange(len(forecast.values)),
            "forecast": forecast.values,
        }
    )
    return df


def convert_backtest_to_flat_forecasts(
    backtest_forecasts: list[BacktestForecast], *, validate: bool = True
) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    total = sum(len(fc.values) for fc in backtest_forecasts)
    loc_col = np.empty(total, dtype=object)
    per_col = np.empty(total, dtype=object)
    hdist_col = np.empty(total, dtype=np.int64)
    sample_col = np.empty(total, dtype=np.int64)
    forecast_col = np.empty(total, dtype=np.float64)

    i = 0
    for fc in backtest_forecasts:
        loc = str(fc.org_unit)
        per = str(fc.period)
        hdist = horizon_diff(per, str(fc.last_seen_period))

        vals = np.asarray(fc.values)
        n = vals.shape[0]

        sl = slice(i, i + n)
        loc_col[sl] = loc
        per_col[sl] = per
        hdist_col[sl] = hdist
        sample_col[sl] = np.arange(n, dtype=np.int64)
        forecast_col[sl] = vals.astype(np.float64, copy=False)

        i += n

    df = pd.DataFrame(
        {
            "location": loc_col,
            "time_period": per_col,
            "horizon_distance": hdist_col,
            "sample": sample_col,
            "forecast": forecast_col,
        }
    )

    # if validate:
    #     FlatForecasts.validate(df)

    return df


def convert_backtest_observations_to_flat_observations(
    observations: list[ObservationBase],
    target_column: str = "disease_cases",
) -> pd.DataFrame:
    """
    Convert a list of ObservationBase objects to a flat DataFrame format
    conforming to ObservedFlatDataSchema.

    Args:
        observations: List of ObservationBase objects containing observations
        reference_period: Optional reference period to calculate horizon_distance from.
                         If provided, horizon_distance will be calculated relative to this.
                         If None, horizon_distance will be set to 0 for all observations.

    Returns:
        pd.DataFrame with columns: location, time_period, disease_cases, plus population
        when the observations include a ``population`` feature
    """
    rows = []
    population: dict[tuple[str, str], float] = {}

    for obs in observations:
        if obs.value is None:
            continue
        key = (str(obs.org_unit), str(obs.period))
        if obs.feature_name == target_column:
            rows.append({"location": key[0], "time_period": key[1], "disease_cases": float(obs.value)})
        elif obs.feature_name == "population":
            population[key] = float(obs.value)

    # Create DataFrame from rows
    df = pd.DataFrame(rows)
    if population and not df.empty:
        df["population"] = [population.get(key) for key in zip(df["location"], df["time_period"], strict=True)]
        df["population"] = df["population"].astype(float)

    if not df.empty:
        # Validate against schema
        FlatObserved.validate(df)

    return df


class DataDimension(StrEnum):
    """
    Enum for the possible dimensions metrics datasets can have
    """

    location = "location"
    time_period = "time_period"
    horizon_distance = "horizon_distance"


# Registry of types for each dimension
DIM_REGISTRY: Mapping[DataDimension, tuple[type, Check | None]] = {
    DataDimension.location: (str, None),
    DataDimension.time_period: (str, None),
    DataDimension.horizon_distance: (int, Check.ge(1)),
}
