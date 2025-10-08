"""
Normalized Continuous Ranked Probability Score (CRPS) metrics.
"""

import pandas as pd
from chap_core.assessment.flat_representations import DataDimension, FlatForecasts, FlatObserved
from chap_core.assessment.metrics.base import MetricBase, MetricSpec
from chap_core.assessment.metrics.crps import DetailedCRPS


class DetailedCRPSNorm(MetricBase):
    """
    Detailed Normalized Continuous Ranked Probability Score (CRPS) metric.
    Does not group - gives one normalized CRPS value per location/time_period/horizon_distance combination.
    CRPS is normalized by the range of observed values to make it comparable across different scales.
    """

    spec = MetricSpec(
        output_dimensions=(DataDimension.location, DataDimension.time_period, DataDimension.horizon_distance),
        metric_name="CRPS Normalized",
        metric_id="detailed_crps_norm",
        description="Normalized CRPS per location, time period and horizon",
    )

    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # First compute regular CRPS for each location/time_period/horizon combination
        detailed_crps_metric = DetailedCRPS()
        detailed_crps_results = detailed_crps_metric.compute(observations, forecasts)

        # Calculate normalization factor based on range of all observed values
        obs_values = observations["disease_cases"].values
        obs_min, obs_max = obs_values.min(), obs_values.max()
        obs_range = obs_max - obs_min

        # Avoid division by zero if all observations are the same
        if obs_range == 0:
            # If all observations are identical, normalized CRPS is just the regular CRPS
            detailed_crps_results["metric"] = detailed_crps_results["metric"]
        else:
            # Normalize CRPS by the range of observations
            detailed_crps_results["metric"] = detailed_crps_results["metric"] / obs_range

        return detailed_crps_results


class CRPSNorm(MetricBase):
    """
    Normalized Continuous Ranked Probability Score (CRPS) metric aggregated by location.
    Groups by location to give average normalized CRPS per location across all time periods and horizons.
    """

    spec = MetricSpec(
        output_dimensions=(DataDimension.location,),
        metric_name="CRPS Normalized",
        metric_id="crps_norm",
        description="Average normalized CRPS per location",
    )

    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # First compute detailed normalized CRPS
        detailed_crps_norm_metric = DetailedCRPSNorm()
        detailed_results = detailed_crps_norm_metric.compute(observations, forecasts)

        # Aggregate by location
        location_crps_norm = detailed_results.groupby("location", as_index=False)["metric"].mean()

        return location_crps_norm
