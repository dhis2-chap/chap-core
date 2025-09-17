from dataclasses import dataclass
from pandera import Column, DataFrameSchema
from chap_core.assessment.flat_representations import DIM_REGISTRY, DataDimension, FlatForecasts, FlatObserved
import pandas as pd


@dataclass(frozen=True)
class MetricSpec:
    group_by: tuple[DataDimension, ...] = ()
    metric_name: str = "metric"


class MetricBase:
    """
    Base class for metrics. Subclass this and implement the compute-method to create a new metric.
    Define the spec attribute to specify what the metric outputs.
    """
    spec: MetricSpec = MetricSpec()

    def get_metric(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        out = self.compute(observations, forecasts)

        expected = [*(d.value for d in self.spec.group_by), self.spec.metric_name]
        missing = [c for c in expected if c not in out.columns]
        extra   = [c for c in out.columns if c not in expected]
        if missing or extra:
            raise ValueError(
                f"{self.__class__.__name__} produced wrong columns.\n"
                f"Expected: {expected}\nMissing: {missing}\nExtra: {extra}"
            )

        return self._make_schema().validate(out, lazy=False)

    def compute(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def _make_schema(self) -> DataFrameSchema:
        cols: dict[str, Column] = {}
        for d in self.spec.group_by:
            dtype, chk = DIM_REGISTRY[d]
            cols[d.value] = Column(dtype, chk) if chk else Column(dtype)
        cols[self.spec.metric_name] = Column(float)
        return DataFrameSchema(cols, strict=True, coerce=True)

    def get_name(self) -> str:
        return self.spec.metric_name


class RMSE(MetricBase):
    """
    Root Mean Squared Error metric.
    Groups by location to give RMSE per location across all time periods and horizons.
    """
    spec = MetricSpec(group_by=(DataDimension.location,))

    def compute(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
        # Merge observations with forecasts on location and time_period
        merged = forecasts.merge(
            observations[['location', 'time_period', 'disease_cases']],
            on=['location', 'time_period'],
            how='inner'
        )

        # Calculate squared error for each forecast
        merged['squared_error'] = (merged['forecast'] - merged['disease_cases']) ** 2

        # First average across samples for each location/time_period combination
        per_sample_mse = (
            merged.groupby(['location', 'time_period', 'sample'], as_index=False)['squared_error']
            .mean()
        )

        # Then average across all time periods and samples for each location
        location_mse = (
            per_sample_mse.groupby('location', as_index=False)['squared_error']
            .mean()
        )

        # Take square root to get RMSE
        location_mse['metric'] = location_mse['squared_error'] ** 0.5

        # Return only the required columns
        return location_mse[['location', 'metric']]


class MAE(MetricBase):
    """
    Mean Absolute Error metric.
    Groups by location and horizon_distance to show error patterns across forecast horizons.
    """
    spec = MetricSpec(group_by=(DataDimension.location, DataDimension.horizon_distance))

    def compute(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
        # Merge observations with forecasts
        merged = forecasts.merge(
            observations[['location', 'time_period', 'disease_cases']],
            on=['location', 'time_period'],
            how='inner'
        )

        # Calculate absolute error
        merged['abs_error'] = (merged['forecast'] - merged['disease_cases']).abs()

        # Average across samples first
        per_sample_mae = (
            merged.groupby(['location', 'horizon_distance', 'sample'], as_index=False)['abs_error']
            .mean()
        )

        # Then average across samples to get MAE per location and horizon
        mae_by_horizon = (
            per_sample_mae.groupby(['location', 'horizon_distance'], as_index=False)['abs_error']
            .mean()
            .rename(columns={'abs_error': 'metric'})
        )

        return mae_by_horizon

    def get_old_format(
class OverallRMSE(MetricBase):