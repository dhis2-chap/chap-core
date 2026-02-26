"""
Outbreak detection metrics: sensitivity and specificity.

An "outbreak" is defined as observed cases exceeding the seasonal baseline
(mean + 2*std) for a given location and calendar month. An "alert" is raised
when >50% of forecast samples exceed this threshold.
"""

import pandas as pd

from chap_core.assessment.flat_representations import FlatObserved
from chap_core.assessment.metrics import metric
from chap_core.assessment.metrics.base import (
    AggregationOp,
    Metric,
    MetricSpec,
)


def compute_seasonal_thresholds(historical_observations: pd.DataFrame) -> pd.DataFrame:
    """Compute outbreak thresholds from historical observations.

    Args:
        historical_observations: DataFrame with columns [location, time_period, disease_cases]

    Returns:
        DataFrame with columns [location, month, threshold]
    """
    df = historical_observations.copy()
    df["month"] = df["time_period"].apply(lambda tp: pd.to_datetime(tp).month)
    grouped = df.groupby(["location", "month"])["disease_cases"].agg(["mean", "std"]).reset_index()
    grouped["threshold"] = grouped["mean"] + 2 * grouped["std"]
    return grouped[["location", "month", "threshold"]]


def _get_thresholds(metric_instance: Metric) -> pd.DataFrame:
    if metric_instance.historical_observations is None:
        return pd.DataFrame(columns=["location", "month", "threshold"])
    return compute_seasonal_thresholds(metric_instance.historical_observations)


def _has_monthly_time_periods(observations: pd.DataFrame | FlatObserved) -> bool:
    """Check if time periods are parseable as dates (monthly format, not weekly like 2022W01)."""
    if observations.empty:
        return False
    sample = str(observations["time_period"].iloc[0])
    try:
        pd.to_datetime(sample)
    except ValueError:
        return False
    return True


@metric()
class SensitivityMetric(Metric):
    """Sensitivity (true positive rate) for outbreak detection.

    Measures the proportion of actual outbreaks that were correctly
    predicted (alerted) by the forecast.
    """

    spec = MetricSpec(
        metric_id="sensitivity",
        metric_name="Sensitivity",
        aggregation_op=AggregationOp.MEAN,
        description="True positive rate for outbreak detection alerts",
    )

    def is_applicable(self, observations: FlatObserved) -> bool:
        return self.historical_observations is not None and _has_monthly_time_periods(observations)

    def compute_detailed(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
        thresholds = _get_thresholds(self)
        empty = pd.DataFrame(columns=["location", "time_period", "horizon_distance", "metric"])
        if thresholds.empty:
            return empty

        obs = observations[["location", "time_period", "disease_cases"]].copy()
        obs["month"] = obs["time_period"].apply(lambda tp: pd.to_datetime(tp).month)
        obs = obs.merge(thresholds, on=["location", "month"], how="left")

        # Keep only condition-positive rows (actual outbreaks)
        obs = obs[obs["disease_cases"] > obs["threshold"]]
        if obs.empty:
            return empty

        # Merge forecasts with thresholds
        fc = forecasts.copy()
        fc["month"] = fc["time_period"].apply(lambda tp: pd.to_datetime(tp).month)
        fc = fc.merge(thresholds, on=["location", "month"], how="left")

        # Compute alert per (location, time_period, horizon_distance)
        fc["exceeds"] = (fc["forecast"] > fc["threshold"]).astype(float)
        alert = fc.groupby(["location", "time_period", "horizon_distance"], as_index=False)["exceeds"].mean()
        alert["metric"] = (alert["exceeds"] > 0.5).astype(float)

        # Join with outbreak rows
        merged = obs[["location", "time_period"]].merge(
            alert[["location", "time_period", "horizon_distance", "metric"]],
            on=["location", "time_period"],
            how="inner",
        )
        return pd.DataFrame(merged[["location", "time_period", "horizon_distance", "metric"]])


@metric()
class SpecificityMetric(Metric):
    """Specificity (true negative rate) for outbreak detection.

    Measures the proportion of non-outbreak periods that were correctly
    not alerted by the forecast.
    """

    spec = MetricSpec(
        metric_id="specificity",
        metric_name="Specificity",
        aggregation_op=AggregationOp.MEAN,
        description="True negative rate for outbreak detection alerts",
    )

    def is_applicable(self, observations: FlatObserved) -> bool:
        return self.historical_observations is not None and _has_monthly_time_periods(observations)

    def compute_detailed(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
        thresholds = _get_thresholds(self)
        empty = pd.DataFrame(columns=["location", "time_period", "horizon_distance", "metric"])
        if thresholds.empty:
            return empty

        obs = observations[["location", "time_period", "disease_cases"]].copy()
        obs["month"] = obs["time_period"].apply(lambda tp: pd.to_datetime(tp).month)
        obs = obs.merge(thresholds, on=["location", "month"], how="left")

        # Keep only condition-negative rows (no outbreak)
        obs = obs[obs["disease_cases"] <= obs["threshold"]]
        if obs.empty:
            return empty

        # Merge forecasts with thresholds
        fc = forecasts.copy()
        fc["month"] = fc["time_period"].apply(lambda tp: pd.to_datetime(tp).month)
        fc = fc.merge(thresholds, on=["location", "month"], how="left")

        # Compute alert per (location, time_period, horizon_distance)
        fc["exceeds"] = (fc["forecast"] > fc["threshold"]).astype(float)
        alert = fc.groupby(["location", "time_period", "horizon_distance"], as_index=False)["exceeds"].mean()
        alert["metric"] = (alert["exceeds"] <= 0.5).astype(float)

        # Join with non-outbreak rows
        merged = obs[["location", "time_period"]].merge(
            alert[["location", "time_period", "horizon_distance", "metric"]],
            on=["location", "time_period"],
            how="inner",
        )
        return pd.DataFrame(merged[["location", "time_period", "horizon_distance", "metric"]])
