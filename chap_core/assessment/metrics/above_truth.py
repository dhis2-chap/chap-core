import pandas as pd
from chap_core.assessment.flat_representations import DataDimension, FlatForecasts, FlatObserved
from chap_core.assessment.metrics.base import MetricBase, MetricSpec

def forecast_mean(forecasts: FlatForecasts) -> pd.DataFrame:
    return (
        forecasts
        .groupby(["location", "time_period", "sample"], as_index=False)["forecast"]
        .mean()
        .rename(columns={"forecast": "forecast_sample_mean"})
    )

class SamplesAboveTruthCountByTime(MetricBase):
    spec = MetricSpec(
        output_dimensions=(DataDimension.location, DataDimension.time_period),
        metric_name="Samples Above Truth (count per time)",
        metric_id="samples_above_truth_count_time",
        description="Number of forecast samples with mean value > truth for each (location, time_period).",
    )

    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        fc = forecast_mean(forecasts)
        obs = observations[["location", "time_period", "disease_cases"]]
        merged = fc.merge(obs, on=["location", "time_period"], how="inner")
        merged["is_above"] = (merged["forecast_sample_mean"] > merged["disease_cases"]).astype(int)
        out = (
            merged.groupby(["location", "time_period"], as_index=False)["is_above"]
            .sum()
            .rename(columns={"is_above": "metric"})
        )

        return out[["location", "time_period", "metric"]]


class SamplesAboveTruthCountByLocation(MetricBase):
    spec = MetricSpec(
        output_dimensions=(DataDimension.location,),
        metric_name="Samples Above Truth (count per location)",
        metric_id="samples_above_truth_count_location",
        description="Total number of forecast samples with mean value > truth across all time periods for each location.",
    )

    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        fc = forecast_mean(forecasts)
        obs = observations[["location", "time_period", "disease_cases"]]
        merged = fc.merge(obs, on=["location", "time_period"], how="inner")
        merged["is_above"] = (merged["forecast_sample_mean"] > merged["disease_cases"]).astype(int)
        out = (
            merged.groupby(["location"], as_index=False)["is_above"]
            .sum()
            .rename(columns={"is_above": "metric"})
        )

        return out[["location", "metric"]]