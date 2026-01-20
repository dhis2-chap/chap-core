"""
Base classes for all metrics.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import pandera.pandas as pa
from chap_core.assessment.flat_representations import (
    DIM_REGISTRY,
    DataDimension,
    FlatForecasts,
    FlatObserved,
)


class AggregationLevel(str, Enum):
    """
    Enum for metric aggregation levels.
    """

    DETAILED = "detailed"  # Per location/time_period/horizon_distance
    PER_LOCATION = "per_location"  # Per location only
    PER_HORIZON = "per_horizon"  # Per horizon_distance only
    AGGREGATE = "aggregate"  # Single scalar


class AggregationOp(str, Enum):
    """
    Enum for aggregation operations.
    """

    MEAN = "mean"
    SUM = "sum"
    ROOT_MEAN_SQUARE = "root_mean_square"


@dataclass(frozen=True)
class UnifiedMetricSpec:
    """
    Specification for a unified metric.
    """

    metric_id: str
    metric_name: str
    aggregation_op: AggregationOp = AggregationOp.MEAN
    description: str = "No description provided"


# Mapping of AggregationLevel to output dimensions
LEVEL_TO_DIMENSIONS: dict[AggregationLevel, tuple[DataDimension, ...]] = {
    AggregationLevel.DETAILED: (
        DataDimension.location,
        DataDimension.time_period,
        DataDimension.horizon_distance,
    ),
    AggregationLevel.PER_LOCATION: (DataDimension.location,),
    AggregationLevel.PER_HORIZON: (DataDimension.horizon_distance,),
    AggregationLevel.AGGREGATE: (),
}


class UnifiedMetric(ABC):
    """
    Base class for unified metrics that support multiple aggregation levels.

    Subclasses implement compute_detailed() which computes the metric at the
    finest resolution. Aggregation to other levels is handled automatically.
    """

    spec: UnifiedMetricSpec

    def get_metric(
        self,
        observations: FlatObserved,
        forecasts: FlatForecasts,
        level: AggregationLevel = AggregationLevel.AGGREGATE,
    ) -> pd.DataFrame:
        """
        Compute the metric at the specified aggregation level.

        Args:
            observations: Flat observations DataFrame
            forecasts: Flat forecasts DataFrame
            level: Aggregation level (DETAILED, PER_LOCATION, PER_HORIZON, AGGREGATE)

        Returns:
            DataFrame with metric values at the specified level
        """
        # Filter out null observations
        null_mask = observations.disease_cases.isnull()
        observations = observations[~null_mask]

        # Compute at detailed level first
        detailed = self.compute_detailed(observations, forecasts)

        # Aggregate to requested level
        result = self._aggregate_to_level(detailed, level)

        # Validate output
        output_dimensions = LEVEL_TO_DIMENSIONS[level]
        return self._validate_output(result, output_dimensions)

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

    def _aggregate_to_level(self, detailed: pd.DataFrame, level: AggregationLevel) -> pd.DataFrame:
        """Aggregate detailed results to the specified level."""
        if level == AggregationLevel.DETAILED:
            return detailed

        dimensions = LEVEL_TO_DIMENSIONS[level]
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
            return grouped.mean()
        elif op == AggregationOp.SUM:
            return grouped.sum()
        elif op == AggregationOp.ROOT_MEAN_SQUARE:
            # For grouped RMS, we need to compute sqrt(mean(x^2)) per group
            # Use agg with a custom function to avoid column renaming issues
            result = grouped.agg(lambda x: np.sqrt((x**2).mean()))
            return result
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

    def get_name(self) -> str:
        """Return the display name of the metric."""
        return self.spec.metric_name

    def get_id(self) -> str:
        """Return the metric ID."""
        return self.spec.metric_id

    def get_description(self) -> str:
        """Return the metric description."""
        return self.spec.description


class DeterministicUnifiedMetric(UnifiedMetric):
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

        return merged[["location", "time_period", "horizon_distance", "metric"]]

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


class ProbabilisticUnifiedMetric(UnifiedMetric):
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
            metric_value = self.compute_sample_metric(sample_values, obs_value)

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
