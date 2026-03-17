"""
Base classes for all metrics.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
import pandera.pandas as pa

from chap_core.assessment.flat_representations import (
    DIM_REGISTRY,
    DataDimension,
    FlatForecasts,
    FlatObserved,
)


class AggregationOp(StrEnum):
    """
    Enum for aggregation operations.
    """

    MEAN = "mean"
    SUM = "sum"
    ROOT_MEAN_SQUARE = "root_mean_square"


# Default output dimensions for detailed metrics
DEFAULT_OUTPUT_DIMENSIONS: tuple[DataDimension, ...] = (
    DataDimension.location,
    DataDimension.time_period,
    DataDimension.horizon_distance,
)


@dataclass(frozen=True)
class MetricSpec:
    """
    Specification for a metric.
    """

    metric_id: str
    metric_name: str
    output_dimensions: tuple[DataDimension, ...] = DEFAULT_OUTPUT_DIMENSIONS
    aggregation_op: AggregationOp = AggregationOp.MEAN
    description: str = "No description provided"


class Metric(ABC):
    """
    Base class for metrics that support multiple aggregation levels.

    Subclasses implement compute_detailed() which computes the metric at the
    finest resolution. Aggregation to other levels is handled automatically.
    """

    spec: MetricSpec

    def __init__(self, historical_observations: pd.DataFrame | None = None):
        self.historical_observations = historical_observations

    def get_global_metric(
        self,
        observations: FlatObserved,
        forecasts: FlatForecasts,
    ) -> pd.DataFrame:
        """
        Compute the metric as a single aggregated scalar value.

        Args:
            observations: Flat observations DataFrame
            forecasts: Flat forecasts DataFrame

        Returns:
            DataFrame with a single metric value
        """
        return self.get_metric(observations, forecasts, dimensions=())

    def get_detailed_metric(
        self,
        observations: FlatObserved,
        forecasts: FlatForecasts,
    ) -> pd.DataFrame:
        """
        Compute the metric at the finest resolution (from spec.output_dimensions).

        Args:
            observations: Flat observations DataFrame
            forecasts: Flat forecasts DataFrame

        Returns:
            DataFrame with metric values at the finest resolution
        """
        return self.get_metric(observations, forecasts, dimensions=self.spec.output_dimensions)

    def get_metric(
        self,
        observations: FlatObserved,
        forecasts: FlatForecasts,
        dimensions: tuple[DataDimension, ...] = (),
    ) -> pd.DataFrame:
        """
        Compute the metric, keeping the specified dimensions.

        Args:
            observations: Flat observations DataFrame
            forecasts: Flat forecasts DataFrame
            dimensions: Which dimensions to keep in the output. Empty tuple means
                       aggregate to a single scalar value.

        Returns:
            DataFrame with metric values at the specified level
        """
        # Filter out null observations
        null_mask = observations.disease_cases.isnull()  # type: ignore[attr-defined]
        observations = observations[~null_mask]  # type: ignore[index]

        # Compute at detailed level first
        detailed = self.compute_detailed(observations, forecasts)  # type: ignore[arg-type]

        # Aggregate to requested level
        result = self._aggregate_to_dimensions(detailed, dimensions)

        # Validate output
        return self._validate_output(result, dimensions)

    @abstractmethod
    def compute_detailed(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the metric at the finest resolution (per location/time_period/horizon_distance).

        Args:
            observations: DataFrame with columns [location, time_period, disease_cases]
            forecasts: DataFrame with columns [location, time_period, horizon_distance, sample, forecast]

        Returns:
            DataFrame with columns [location, time_period, horizon_distance, metric]
        """
        raise NotImplementedError

    def _aggregate_to_dimensions(self, detailed: pd.DataFrame, dimensions: tuple[DataDimension, ...]) -> pd.DataFrame:
        """Aggregate detailed results to keep only the specified dimensions."""
        # Check if dimensions match the detailed output (no aggregation needed)
        if dimensions == self.spec.output_dimensions:
            return detailed

        group_cols = [d.value for d in dimensions]

        if not group_cols:
            # Full aggregate - no grouping
            return self._apply_aggregation(detailed["metric"])

        # Group by specified dimensions
        grouped = detailed.groupby(group_cols, as_index=False)["metric"]
        return self._apply_grouped_aggregation(grouped, group_cols)

    def _apply_aggregation(self, series: pd.Series) -> pd.DataFrame:
        """Apply aggregation operation to a series, returning a single-row DataFrame."""
        op = self.spec.aggregation_op
        if op == AggregationOp.MEAN:
            value = series.mean()
        elif op == AggregationOp.SUM:
            value = series.sum()
        elif op == AggregationOp.ROOT_MEAN_SQUARE:
            value = np.sqrt((series**2).mean())
        else:
            raise ValueError(f"Unknown aggregation operation: {op}")
        return pd.DataFrame({"metric": [value]})

    def _apply_grouped_aggregation(self, grouped, group_cols: list[str]) -> pd.DataFrame:
        """Apply aggregation operation to grouped data."""
        op = self.spec.aggregation_op
        if op == AggregationOp.MEAN:
            return pd.DataFrame(grouped.mean())
        elif op == AggregationOp.SUM:
            return pd.DataFrame(grouped.sum())
        elif op == AggregationOp.ROOT_MEAN_SQUARE:
            # For grouped RMS, we need to compute sqrt(mean(x^2)) per group
            # Use agg with a custom function to avoid column renaming issues
            result = grouped.agg(lambda x: np.sqrt((x**2).mean()))
            return pd.DataFrame(result)
        else:
            raise ValueError(f"Unknown aggregation operation: {op}")

    def _validate_output(self, result: pd.DataFrame, output_dimensions: tuple[DataDimension, ...]) -> pd.DataFrame:
        """Validate and coerce output DataFrame."""
        expected = [*(d.value for d in output_dimensions), "metric"]
        missing = [c for c in expected if c not in result.columns]
        extra = [c for c in result.columns if c not in expected]
        if missing or extra:
            raise ValueError(
                f"{self.__class__.__name__} produced wrong columns.\n"
                f"Expected: {expected}\nMissing: {missing}\nExtra: {extra}"
            )
        return self._make_schema(output_dimensions).validate(result, lazy=False)

    def _make_schema(self, output_dimensions: tuple[DataDimension, ...]) -> pa.DataFrameSchema:
        """Create a pandera schema for validation."""
        cols: dict[str, pa.Column] = {}
        for d in output_dimensions:
            dtype, chk = DIM_REGISTRY[d]
            cols[d.value] = pa.Column(dtype, chk) if chk else pa.Column(dtype)
        cols["metric"] = pa.Column(float, nullable=True)
        return pa.DataFrameSchema(cols, strict=True, coerce=True)

    def is_applicable(self, observations: FlatObserved) -> bool:
        """Check whether this metric can be computed for the given data.

        Subclasses may override to indicate they require specific data
        (e.g. historical observations or a particular time period format).
        Callers should skip metrics that return False.
        """
        return True

    def get_name(self) -> str:
        """Return the display name of the metric."""
        return self.spec.metric_name

    def get_id(self) -> str:
        """Return the metric ID."""
        return self.spec.metric_id

    def get_description(self) -> str:
        """Return the metric description."""
        return self.spec.description


class DeterministicMetric(Metric):
    """
    Base class for deterministic metrics that operate on the median of samples.

    Subclasses implement compute_point_metric() which receives a forecast value
    and observed value and returns the metric value.
    """

    def compute_detailed(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
        """Compute detailed metric using median of samples."""
        # Compute median forecast across samples
        median_forecasts = forecasts.groupby(["location", "time_period", "horizon_distance"], as_index=False)[
            "forecast"
        ].median()

        # Merge with observations
        merged = median_forecasts.merge(
            observations[["location", "time_period", "disease_cases"]],
            on=["location", "time_period"],
            how="inner",
        )

        # Compute metric for each row
        merged["metric"] = merged.apply(
            lambda row: self.compute_point_metric(row["forecast"], row["disease_cases"]), axis=1
        )

        return pd.DataFrame(merged[["location", "time_period", "horizon_distance", "metric"]])

    @abstractmethod
    def compute_point_metric(self, forecast: float, observed: float) -> float:
        """
        Compute the metric for a single forecast/observation pair.

        Args:
            forecast: The median forecast value
            observed: The observed value

        Returns:
            The metric value
        """
        raise NotImplementedError


class ProbabilisticMetric(Metric):
    """
    Base class for probabilistic metrics that need all samples.

    Subclasses implement compute_sample_metric() which receives all sample
    values and the observed value and returns the metric value.
    """

    def compute_detailed(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
        """Compute detailed metric using all samples."""
        # Merge observations with forecasts on location and time_period
        merged = forecasts.merge(
            observations[["location", "time_period", "disease_cases"]], on=["location", "time_period"], how="inner"
        )

        # Group by location, time_period, and horizon_distance to compute metric
        results = []
        for (location, time_period, horizon), group in merged.groupby(["location", "time_period", "horizon_distance"]):
            # Get all sample values for this combination
            sample_values = group["forecast"].values
            # Get the observation (should be the same for all samples)
            obs_value = group["disease_cases"].iloc[0]

            # Compute the metric
            metric_value = self.compute_sample_metric(np.asarray(sample_values), obs_value)

            results.append(
                {"location": location, "time_period": time_period, "horizon_distance": horizon, "metric": metric_value}
            )

        return pd.DataFrame(results)

    @abstractmethod
    def compute_sample_metric(self, samples: np.ndarray, observed: float) -> float:
        """
        Compute the metric from all samples and the observation.

        Args:
            samples: Array of all forecast sample values
            observed: The observed value

        Returns:
            The metric value
        """
        raise NotImplementedError
