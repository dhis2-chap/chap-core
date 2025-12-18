"""
Base classes for all metrics.
"""

from dataclasses import dataclass
import pandas as pd
import pandera.pandas as pa
from chap_core.assessment.flat_representations import (
    DIM_REGISTRY,
    DataDimension,
    FlatForecasts,
    FlatObserved,
)


@dataclass(frozen=True)
class MetricSpec:
    output_dimensions: tuple[DataDimension, ...] = ()
    metric_name: str = "metric"
    metric_id: str = "metric"
    description: str = "No description provided"


class MetricBase:
    """
    Base class for metrics. Subclass this and implement the compute-method to create a new metric.
    Define the spec attribute to specify what the metric outputs.
    """

    spec: MetricSpec = MetricSpec()

    def get_metric(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # Check taht obserations ar not nan
        null_mask = observations.disease_cases.isnull()
        observations = observations[~null_mask]
        out = self.compute(observations, forecasts)

        expected = [*(d for d in self.spec.output_dimensions), "metric"]
        missing = [c for c in expected if c not in out.columns]
        extra = [c for c in out.columns if c not in expected]
        if missing or extra:
            raise ValueError(
                f"{self.__class__.__name__} produced wrong columns.\n"
                f"Expected: {expected}\nMissing: {missing}\nExtra: {extra}"
            )

        return self._make_schema().validate(out, lazy=False)

    def compute(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def _make_schema(self) -> pa.DataFrameSchema:
        cols: dict[str, pa.Column] = {}
        for d in self.spec.output_dimensions:
            dtype, chk = DIM_REGISTRY[d]
            cols[d.value] = pa.Column(dtype, chk) if chk else pa.Column(dtype)
        cols["metric"] = pa.Column(float, nullable=True)
        return pa.DataFrameSchema(cols, strict=True, coerce=True)

    def get_name(self) -> str:
        return self.spec.metric_name

    def gives_highest_resolution(self) -> bool:
        """
        Returns True if the metric gives one number per location/time_period/horizon_distance combination.
        """
        return len(self.spec.output_dimensions) == 3

    def is_full_aggregate(self) -> bool:
        """
        Returns True if the metric gives only one number for the whole dataset
        """
        return len(self.spec.output_dimensions) == 0


class DeterministicMetric(MetricBase):
    """
    Base class for deterministic metrics that operate on the median of samples.
    Subclasses implement compute_from_merged() which receives a DataFrame with:
    - location, time_period, horizon_distance (from forecasts)
    - forecast (median across samples)
    - disease_cases (from observations)
    """

    def compute(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
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

        return self.compute_from_merged(merged)

    def compute_from_merged(self, merged: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the metric from the merged DataFrame.

        Args:
            merged: DataFrame with columns [location, time_period, horizon_distance, forecast, disease_cases]

        Returns:
            DataFrame with columns matching spec.output_dimensions + ["metric"]
        """
        raise NotImplementedError
