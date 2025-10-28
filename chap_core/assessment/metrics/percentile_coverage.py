"""
Percentile coverage metrics for evaluating forecast calibration.
"""

import numpy as np
import pandas as pd
from chap_core.assessment.flat_representations import DataDimension, FlatForecasts, FlatObserved
from chap_core.assessment.metrics.base import MetricBase, MetricSpec


class IsWithin10th90thDetailed(MetricBase):
    """
    Detailed metric checking if observation falls within 10th-90th percentile of forecast samples.
    Does not group - gives one binary value (0 or 1) per location/time_period/horizon_distance combination.
    Returns 1 if observation is within the 10th-90th percentile range, 0 otherwise.
    """

    spec = MetricSpec(
        output_dimensions=(DataDimension.location, DataDimension.time_period, DataDimension.horizon_distance),
        metric_name="Within 10-90 Percentile",
        metric_id="is_within_10th_90th_detailed",
        description="Binary indicator if observation is within 10th-90th percentile per location, time period and horizon",
    )

    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # Merge observations with forecasts on location and time_period
        merged = forecasts.merge(
            observations[["location", "time_period", "disease_cases"]], on=["location", "time_period"], how="inner"
        )

        # Group by location, time_period, and horizon_distance to compute percentile coverage
        results = []
        for (location, time_period, horizon), group in merged.groupby(["location", "time_period", "horizon_distance"]):
            # Get all sample values for this combination
            sample_values = group["forecast"].values
            # Get the observation (should be the same for all samples)
            obs_value = group["disease_cases"].iloc[0]

            # Calculate 10th and 90th percentiles of the samples
            low, high = np.percentile(sample_values, [10, 90])
            # Check if observation falls within this range
            is_within_range = 1.0 if (low <= obs_value <= high) else 0.0

            results.append(
                {
                    "location": location,
                    "time_period": time_period,
                    "horizon_distance": horizon,
                    "metric": is_within_range,
                }
            )

        return pd.DataFrame(results)


class IsWithin25th75thDetailed(MetricBase):
    """
    Detailed metric checking if observation falls within 25th-75th percentile of forecast samples.
    Does not group - gives one binary value (0 or 1) per location/time_period/horizon_distance combination.
    Returns 1 if observation is within the 25th-75th percentile range, 0 otherwise.
    """

    spec = MetricSpec(
        output_dimensions=(DataDimension.location, DataDimension.time_period, DataDimension.horizon_distance),
        metric_name="Within 25-75 Percentile",
        metric_id="is_within_25th_75th_detailed",
        description="Binary indicator if observation is within 25th-75th percentile",
    )

    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # Merge observations with forecasts on location and time_period
        merged = forecasts.merge(
            observations[["location", "time_period", "disease_cases"]], on=["location", "time_period"], how="inner"
        )

        # Group by location, time_period, and horizon_distance to compute percentile coverage
        results = []
        for (location, time_period, horizon), group in merged.groupby(["location", "time_period", "horizon_distance"]):
            # Get all sample values for this combination
            sample_values = group["forecast"].values
            # Get the observation (should be the same for all samples)
            obs_value = group["disease_cases"].iloc[0]

            # Calculate 25th and 75th percentiles of the samples
            low, high = np.percentile(sample_values, [25, 75])
            # Check if observation falls within this range
            is_within_range = 1.0 if (low <= obs_value <= high) else 0.0

            results.append(
                {
                    "location": location,
                    "time_period": time_period,
                    "horizon_distance": horizon,
                    "metric": is_within_range,
                }
            )

        return pd.DataFrame(results)


class RatioWithin10th90thPerLocation(MetricBase):
    """
    Ratio of observations within 10th-90th percentile, aggregated by location.
    Groups by location to give the proportion of forecasts where observation fell within range.
    """

    spec = MetricSpec(
        output_dimensions=(DataDimension.location,),
        metric_name="Ratio Within 10-90 Percentile",
        metric_id="ratio_within_10th_90th_per_location",
        description="Ratio of observations within 10th-90th percentile per location",
    )

    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # First compute detailed metric
        detailed_metric = IsWithin10th90thDetailed()
        detailed_results = detailed_metric.compute(observations, forecasts)

        # Aggregate by location (mean of binary values gives ratio)
        location_ratios = detailed_results.groupby("location", as_index=False)["metric"].mean()

        return location_ratios


class RatioWithin10th90th(MetricBase):
    """
    Overall ratio of observations within 10th-90th percentile for entire dataset.
    Gives one ratio value across all locations, time periods and horizons.
    """

    spec = MetricSpec(
        output_dimensions=(),
        metric_name="Ratio Within 10-90 Percentile",
        metric_id="ratio_within_10th_90th",
        description="Overall ratio of observations within 10th-90th percentile",
    )

    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # First compute ratio per location
        ratio_per_location_metric = RatioWithin10th90thPerLocation()
        location_results = ratio_per_location_metric.compute(observations, forecasts)

        # Aggregate across all locations to get overall ratio
        overall_ratio = location_results["metric"].mean()

        return pd.DataFrame({"metric": [overall_ratio]})


class RatioWithin25th75thPerLocation(MetricBase):
    """
    Ratio of observations within 25th-75th percentile, aggregated by location.
    Groups by location to give the proportion of forecasts where observation fell within range.
    """

    spec = MetricSpec(
        output_dimensions=(DataDimension.location,),
        metric_name="Ratio Within 25-75 Percentile",
        metric_id="ratio_within_25th_75th_per_location",
        description="Ratio of observations within 25th-75th percentile per location",
    )

    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # First compute detailed metric
        detailed_metric = IsWithin25th75thDetailed()
        detailed_results = detailed_metric.compute(observations, forecasts)

        # Aggregate by location (mean of binary values gives ratio)
        location_ratios = detailed_results.groupby("location", as_index=False)["metric"].mean()

        return location_ratios


class RatioWithin25th75th(MetricBase):
    """
    Overall ratio of observations within 25th-75th percentile for entire dataset.
    Gives one ratio value across all locations, time periods and horizons.
    """

    spec = MetricSpec(
        output_dimensions=(),
        metric_name="Ratio Within 25-75 Percentile",
        metric_id="ratio_within_25th_75th",
        description="Overall ratio of observations within 25th-75th percentile",
    )

    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # First compute ratio per location
        ratio_per_location_metric = RatioWithin25th75thPerLocation()
        location_results = ratio_per_location_metric.compute(observations, forecasts)

        # Aggregate across all locations to get overall ratio
        overall_ratio = location_results["metric"].mean()

        return pd.DataFrame({"metric": [overall_ratio]})
