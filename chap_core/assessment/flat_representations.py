from __future__ import annotations
from enum import Enum
from typing import Mapping, List

import numpy as np
import pandas as pd
from pandera import Check
import pandera.pandas as pa
from pandera.pandas import DataFrameModel

from chap_core.database.tables import BackTestForecast
from chap_core.database.dataset_tables import ObservationBase
from chap_core.time_period import TimePeriod


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
    Observed disease cases
    """

    disease_cases: pa.typing.Series[float] = pa.Field(nullable=True)  # float to also allow nan


class FlatForecasts(FlatDataWithHorizon):
    """
    Forecasted disease cases. Note that cases are in forecast field, and that samples is used
    so we can represent multiple samples per location/time_period/horizon_distance in the dataframe.
    """

    sample: pa.typing.Series[int]  # index of sample
    forecast: pa.typing.Series[float]  # actual forecast value


class FlatMetric(FlatDataWithHorizon):
    metric: pa.typing.Series[float] = pa.Field(nullable=True)


def horizon_diff(period: str, period2: str) -> int:
    """Calculate the difference between two time periods in terms of time units."""
    tp = TimePeriod.parse(period)
    tp2 = TimePeriod.parse(period2)
    return (tp - tp2) // tp.time_delta


def _convert_backtest_to_flat_forecasts(backtest_forecasts: List[BackTestForecast]) -> pd.DataFrame:
    """
    Convert a list of BackTestForecast objects to a flat DataFrame format
    conforming to ForecastFlatDataSchema.

    Args:
        backtest_forecasts: List of BackTestForecast objects containing forecasts

    Returns:
        pd.DataFrame with columns: location, time_period, horizon_distance, sample, forecast
    """
    dfs = []

    for forecast in backtest_forecasts:
        # Calculate horizon distance using the horizon_diff function
        # horizon_distance represents how many time periods ahead this forecast is
        # from the last period we had data for
        horizon_distance = horizon_diff(str(forecast.period), str(forecast.last_seen_period))

        # Each BackTestForecast contains multiple sample values
        # We need to create one row per sample

        df = _create_df(forecast, horizon_distance)
        dfs.append(df)
        """
        for sample_idx, sample_value in enumerate(forecast.values):
            assert not np.isnan(sample_value), ("Sample values should not be NaN. "
                                                "Potentially something wrong with forecasts given by model")
            row = {
                "location": str(forecast.org_unit),
                "time_period": str(forecast.period),
                "horizon_distance": horizon_distance,
                "sample": sample_idx,
                "forecast": float(sample_value),  # Convert to int as per schema
            }
            rows.append(row)
        """

    # Create DataFrame from rows
    # df = pd.DataFrame(rows)
    df = pd.concat(dfs, ignore_index=True)

    assert len(df) > 0, "No forecast data found in backtest forecasts. Something wrong in model?"

    # Validate against schema
    # FlatForecasts.validate(df)

    return df


def _create_df(forecast: BackTestForecast, horizon_distance: int):
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
    backtest_forecasts: List[BackTestForecast], *, validate: bool = True
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
    observations: List[ObservationBase],
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
        pd.DataFrame with columns: location, time_period, horizon_distance, disease_cases
    """
    rows = []

    for obs in observations:
        # Only process disease_cases observations
        if obs.feature_name == "disease_cases" and obs.value is not None:
            # Calculate horizon distance if reference period is provided
            row = {"location": str(obs.org_unit), "time_period": str(obs.period), "disease_cases": float(obs.value)}
            rows.append(row)

    # Create DataFrame from rows
    df = pd.DataFrame(rows)

    if not df.empty:
        # Validate against schema
        FlatObserved.validate(df)

    return df


def group_flat_forecast_by_horizon(flat_forecast_df: pd.DataFrame, aggregate_samples: bool = True) -> pd.DataFrame:
    """
    Group flat forecast data by horizon distance for analysis.

    Args:
        flat_forecast_df: DataFrame conforming to ForecastFlatDataSchema
        aggregate_samples: If True, average across samples to get mean forecast

    Returns:
        pd.DataFrame grouped by location and horizon_distance
    """
    if aggregate_samples:
        # Average across samples to get mean forecast per location/time_period/horizon
        grouped = flat_forecast_df.groupby(["location", "time_period", "horizon_distance"], as_index=False)[
            "forecast"
        ].mean()
    else:
        grouped = flat_forecast_df

    return grouped


class DataDimension(str, Enum):
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
    DataDimension.horizon_distance: (int, Check.ge(0)),
}
