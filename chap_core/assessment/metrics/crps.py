"""
Continuous Ranked Probability Score (CRPS) metrics.
"""

import numpy as np
import pandas as pd
from chap_core.assessment.flat_representations import DataDimension, FlatForecasts, FlatObserved
from chap_core.assessment.metrics.base import MetricBase, MetricSpec


class DetailedCRPS(MetricBase):
    """
    Detailed Continuous Ranked Probability Score (CRPS) metric.
    Does not group - gives one CRPS value per location/time_period/horizon_distance combination.
    CRPS measures both calibration and sharpness of probabilistic forecasts.
    """

    spec = MetricSpec(
        output_dimensions=(DataDimension.location, DataDimension.time_period, DataDimension.horizon_distance),
        metric_name="CRPS",
        metric_id="detailed_crps",
        description="CRPS per location, time period and horizon",
    )

    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # Merge observations with forecasts on location and time_period
        merged = forecasts.merge(
            observations[["location", "time_period", "disease_cases"]], on=["location", "time_period"], how="inner"
        )

        # Group by location, time_period, and horizon_distance to compute CRPS
        results = []
        for (location, time_period, horizon), group in merged.groupby(["location", "time_period", "horizon_distance"]):
            # Get all sample values for this combination
            sample_values = group["forecast"].values
            # Get the observation (should be the same for all samples)
            obs_value = group["disease_cases"].iloc[0]

            # Calculate CRPS using the formula from database.py
            # CRPS = E[|X - obs|] - 0.5 * E[|X - X'|]
            term1 = np.mean(np.abs(sample_values - obs_value))
            term2 = 0.5 * np.mean(np.abs(sample_values[:, None] - sample_values[None, :]))
            crps = float(term1 - term2)

            results.append(
                {"location": location, "time_period": time_period, "horizon_distance": horizon, "metric": crps}
            )

        return pd.DataFrame(results)


class CRPSPerLocation(MetricBase):
    """
    Continuous Ranked Probability Score (CRPS) metric aggregated by location.
    Groups by location to give average CRPS per location across all time periods and horizons.
    """

    spec = MetricSpec(
        output_dimensions=(DataDimension.location,),
        metric_name="CRPS",
        metric_id="crps_per_location",
        description="Average CRPS per location",
    )

    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # First compute detailed CRPS
        detailed_crps_metric = DetailedCRPS()
        detailed_results = detailed_crps_metric.compute(observations, forecasts)

        # Aggregate by location
        location_crps = detailed_results.groupby("location", as_index=False)["metric"].mean()

        return location_crps


class CRPS(MetricBase):
    """
    Continuous Ranked Probability Score (CRPS) metric for the entire dataset.
    Gives one CRPS value across all locations, time periods and horizons.
    """

    spec = MetricSpec(
        output_dimensions=(), metric_name="CRPS", metric_id="crps", description="Overall CRPS across entire dataset"
    )

    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # First compute CRPS per location
        crps_per_location_metric = CRPSPerLocation()
        location_results = crps_per_location_metric.compute(observations, forecasts)

        # Aggregate across all locations to get overall CRPS
        overall_crps = location_results["metric"].mean()

        return pd.DataFrame({"metric": [overall_crps]})
