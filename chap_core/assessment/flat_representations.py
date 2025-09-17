from __future__ import annotations
from enum import Enum
from typing import Mapping, List

import pandas as pd
import pandera as pa
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
    disease_cases: pa.typing.Series[int]


class FlatForecasts(FlatDataWithHorizon):
    """
    Forecasted disease cases. Note that cases are in forecast field, and that samples is used
    so we can represent multiple samples per location/time_period/horizon_distance in the dataframe. 
    """
    sample: pa.typing.Series[int]  # index of sample
    forecast: pa.typing.Series[int]  # actual forecast value


class FlatMetric(FlatDataWithHorizon):
    metric: pa.typing.Series[float]


def horizon_diff(period: str, period2: str) -> int:
    """Calculate the difference between two time periods in terms of time units."""
    tp = TimePeriod.parse(period)
    tp2 = TimePeriod.parse(period2)
    return (tp - tp2) // tp.time_delta


def convert_backtest_to_flat_forecasts(backtest_forecasts: List[BackTestForecast]) -> pd.DataFrame:
    """
    Convert a list of BackTestForecast objects to a flat DataFrame format
    conforming to ForecastFlatDataSchema.
    
    Args:
        backtest_forecasts: List of BackTestForecast objects containing forecasts
        
    Returns:
        pd.DataFrame with columns: location, time_period, horizon_distance, sample, forecast
    """
    rows = []
    
    for forecast in backtest_forecasts:
        # Calculate horizon distance using the horizon_diff function
        # horizon_distance represents how many time periods ahead this forecast is 
        # from the last period we had data for
        horizon_distance = horizon_diff(str(forecast.period), str(forecast.last_seen_period))
        
        # Each BackTestForecast contains multiple sample values
        # We need to create one row per sample
        for sample_idx, sample_value in enumerate(forecast.values):
            row = {
                'location': str(forecast.org_unit),
                'time_period': str(forecast.period),
                'horizon_distance': horizon_distance,
                'sample': sample_idx,
                'forecast': int(sample_value)  # Convert to int as per schema
            }
            rows.append(row)
    
    # Create DataFrame from rows
    df = pd.DataFrame(rows)
    
    # Validate against schema
    FlatForecasts.validate(df)
    
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
            row = {
                'location': str(obs.org_unit),
                'time_period': str(obs.period),
                'disease_cases': int(obs.value)  # Convert to int as per schema
            }
            rows.append(row)
    
    # Create DataFrame from rows
    df = pd.DataFrame(rows)
    
    if not df.empty:
        # Validate against schema
        FlatObserved.validate(df)
    
    return df


def group_flat_forecast_by_horizon(
    flat_forecast_df: pd.DataFrame,
    aggregate_samples: bool = True
) -> pd.DataFrame:
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
        grouped = flat_forecast_df.groupby(
            ['location', 'time_period', 'horizon_distance'], 
            as_index=False
        )['forecast'].mean()
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







