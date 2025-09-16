from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, List

import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema
import pandera.pandas as pa
from pandera.pandas import DataFrameModel

from chap_core.database.tables import BackTestForecast
from chap_core.database.dataset_tables import ObservationBase
from chap_core.time_period import TimePeriod

# -------------------------------------------------------------------
# 1) Dimensions and global registry (implementers never touch this)
# -------------------------------------------------------------------

class Dim(str, Enum):
    location = "location"
    time_period = "time_period"
    horizon_distance = "horizon_distance"

# Central truth: dtype + optional checks
DIM_REGISTRY: Mapping[Dim, tuple[type, Check | None]] = {
    Dim.location: (str, None),
    Dim.time_period: (str, None),  # could enforce ISO date format with regex
    Dim.horizon_distance: (int, Check.ge(0)),
}


@dataclass(frozen=True)
class MetricSpec:
    group_by: tuple[Dim, ...] = ()
    metric_name: str = "metric"  # always the same


class MetricBase:
    spec: MetricSpec = MetricSpec()

    alignment_keys: tuple[str, ...] = tuple(d.value for d in Dim)

    def compute(self, obs: pd.DataFrame, fcst: pd.DataFrame) -> pd.DataFrame:
        """Public API: run _compute and validate result schema."""
        out = self._compute(obs, fcst)

        # expected columns = group_by dims + "metric"
        expected_cols = [*(d.value for d in self.spec.group_by), self.spec.metric_name]
        missing = [c for c in expected_cols if c not in out.columns]
        extra = [c for c in out.columns if c not in expected_cols]
        if missing or extra:
            raise ValueError(
                f"{self.__class__.__name__} produced wrong columns.\n"
                f"Expected: {expected_cols}\n"
                f"Missing: {missing}\n"
                f"Extra: {extra}"
            )

        return self._make_schema().validate(out)

    def _compute(self, obs: pd.DataFrame, fcst: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def _make_schema(self) -> DataFrameSchema:
        cols: dict[str, Column] = {}

        # group-by dims
        for d in self.spec.group_by:
            dtype, chk = DIM_REGISTRY[d]
            cols[d.value] = Column(dtype, chk) if chk else Column(dtype)

        # metric column: always float, non-negative check optional
        cols[self.spec.metric_name] = Column(float)

        return DataFrameSchema(cols, strict=True, coerce=True)


class MAE(MetricBase):
    spec = MetricSpec(group_by=(Dim.location, Dim.horizon_distance))

    def _compute(self, obs: pd.DataFrame, fcst: pd.DataFrame) -> pd.DataFrame:
        keys = list(self.alignment_keys)
        merged = fcst.merge(
            obs[keys + ["disease_cases"]],
            on=keys,
            how="inner",
            validate="many_to_one",
        )
        merged["abs_err"] = (merged["forecast"] - merged["disease_cases"]).abs()

        # per-sample mean
        per_sample = (
            merged.groupby(keys + ["sample"], as_index=False, dropna=False)["abs_err"]
                  .mean()
        )

        keep = [d.value for d in self.spec.group_by]
        if keep:
            out = (
                per_sample.groupby(keep, as_index=False, dropna=False)["abs_err"]
                          .mean()
                          .rename(columns={"abs_err": "metric"})
            )
        else:
            out = pd.DataFrame({"metric": [per_sample["abs_err"].mean()]})

        return out


class RMSE(MetricBase):
    spec = MetricSpec(group_by=(Dim.location,))

    def _compute(self, obs: pd.DataFrame, fcst: pd.DataFrame) -> pd.DataFrame:
        keys = list(self.alignment_keys)
        merged = fcst.merge(
            obs[keys + ["disease_cases"]],
            on=keys,
            how="inner",
            validate="many_to_one",
        )
        merged["sq_err"] = (merged["forecast"] - merged["disease_cases"]) ** 2

        per_sample = (
            merged.groupby(keys + ["sample"], as_index=False, dropna=False)["sq_err"]
                  .mean()
        )

        keep = [d.value for d in self.spec.group_by]
        if keep:
            out = (
                per_sample.groupby(keep, as_index=False, dropna=False)["sq_err"]
                          .mean()
                          .apply(lambda s: s**0.5)
                          .reset_index()
                          .rename(columns={"sq_err": "metric"})
            )
        else:
            out = pd.DataFrame({"metric": [per_sample["sq_err"].mean() ** 0.5]})

        return out


class BaseFlatDataSchema(DataFrameModel):
    location: pa.typing.Series[str]
    time_period: pa.typing.Series[str]
    horizon_distance: pa.typing.Series[int]


class ObservedFlatDataSchema(BaseFlatDataSchema):
    disease_cases: pa.typing.Series[int]


class ForecastFlatDataSchema(BaseFlatDataSchema):
    sample: pa.typing.Series[int]  # index of sample
    forecast: pa.typing.Series[int]  # actual forecast value


def horizon_diff(period: str, period2: str) -> int:
    """Calculate the difference between two time periods in terms of time units."""
    tp = TimePeriod.parse(period)
    tp2 = TimePeriod.parse(period2)
    return (tp - tp2) // tp.time_delta


def convert_to_forecast_flat_data(backtest_forecasts: List[BackTestForecast]) -> pd.DataFrame:
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
    ForecastFlatDataSchema.validate(df)
    
    return df


def convert_to_forecast_flat_data_with_validation(
    backtest_forecasts: List[BackTestForecast]
) -> pd.DataFrame:
    """
    Convert a list of BackTestForecast objects to a flat DataFrame format
    with additional validation and error handling.
    
    This is a wrapper around convert_to_forecast_flat_data that adds
    additional checks and error handling.
    
    Args:
        backtest_forecasts: List of BackTestForecast objects
        
    Returns:
        pd.DataFrame conforming to ForecastFlatDataSchema
        
    Raises:
        ValueError: If the input is empty or contains invalid data
    """
    if not backtest_forecasts:
        raise ValueError("Input list of BackTestForecast objects cannot be empty")
    
    # Check that all forecasts have values
    for i, forecast in enumerate(backtest_forecasts):
        if not forecast.values:
            raise ValueError(f"BackTestForecast at index {i} has no sample values")
        if forecast.period is None:
            raise ValueError(f"BackTestForecast at index {i} has no period")
        if forecast.last_seen_period is None:
            raise ValueError(f"BackTestForecast at index {i} has no last_seen_period")
        if forecast.org_unit is None:
            raise ValueError(f"BackTestForecast at index {i} has no org_unit")
    
    return convert_to_forecast_flat_data(backtest_forecasts)


def convert_observations_to_flat_data(
    observations: List[ObservationBase],
    reference_period: str = None
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
            if reference_period:
                horizon_distance = horizon_diff(str(obs.period), reference_period)
            else:
                horizon_distance = 0
            
            row = {
                'location': str(obs.org_unit),
                'time_period': str(obs.period),
                'horizon_distance': horizon_distance,
                'disease_cases': int(obs.value)  # Convert to int as per schema
            }
            rows.append(row)
    
    # Create DataFrame from rows
    df = pd.DataFrame(rows)
    
    if not df.empty:
        # Validate against schema
        ObservedFlatDataSchema.validate(df)
    
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

