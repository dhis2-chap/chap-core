import pandas as pd
from chap_core.assessment.flat_representations import DataDimension, FlatForecasts, FlatObserved
from chap_core.assessment.metrics.base import MetricBase, MetricSpec


class SamplesAboveTruth(MetricBase):
    spec = MetricSpec(
        output_dimensions=(DataDimension.location, DataDimension.time_period, DataDimension.horizon_distance),
        metric_name="Samples Above Truth (per time & horizon)",
        metric_id="samples_above_truth_count_time_hz",
        description="Count of forecast samples with mean value > truth for each (location, time_period, horizon_distance).",
    )

    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # average within-sample (if duplicates)
        fc = (
            forecasts.groupby(["location", "time_period", "horizon_distance", "sample"], as_index=False)["forecast"]
            .mean()
            .rename(columns={"forecast": "forecast_sample_mean"})
        )
        obs = observations[["location", "time_period", "disease_cases"]]
        merged = fc.merge(obs, on=["location", "time_period"], how="inner")
        merged["is_above"] = (merged["forecast_sample_mean"] > merged["disease_cases"]).astype(int)

        out = (
            merged.groupby(["location", "time_period", "horizon_distance"], as_index=False)["is_above"]
            .sum()
            .rename(columns={"is_above": "metric"})
        )
        out["metric"] = out["metric"].astype("float64")
        return out[["location", "time_period", "horizon_distance", "metric"]]


class RatioOfSamplesAboveTruth(MetricBase):
    spec = MetricSpec(
        output_dimensions=(DataDimension.location, DataDimension.time_period, DataDimension.horizon_distance),
        metric_name="Ratio of Samples Above Truth (per time & horizon)",
        metric_id="ratio_samples_above_truth_time_hz",
        description="Ratio (0.0-1.0) of forecast samples with mean value > truth for each (location, time_period, horizon_distance).",
    )

    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # average within-sample (if duplicates)
        fc = (
            forecasts.groupby(["location", "time_period", "horizon_distance", "sample"], as_index=False)["forecast"]
            .mean()
            .rename(columns={"forecast": "forecast_sample_mean"})
        )
        obs = observations[["location", "time_period", "disease_cases"]]
        merged = fc.merge(obs, on=["location", "time_period"], how="inner")
        merged["is_above"] = (merged["forecast_sample_mean"] > merged["disease_cases"]).astype(int)

        # Calculate ratio: sum of samples above truth / total count of samples
        agg = merged.groupby(["location", "time_period", "horizon_distance"], as_index=False)["is_above"].agg(
            ["sum", "count"]
        )
        agg["metric"] = agg["sum"] / agg["count"]

        # Reset index to get grouping columns back
        out = agg.reset_index()
        out["metric"] = out["metric"].astype("float64")

        return out[["location", "time_period", "horizon_distance", "metric"]]
