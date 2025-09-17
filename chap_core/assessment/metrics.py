"""
This file contains metric implementations that are based on "flat" datastructures (from flat_representations.py).

New metrics can be implemented in this file by sublacscing MetricBase and implementing the compute-method.

Metrics that should be available should be added to the available_metrics dict.

"""


import logging
from dataclasses import dataclass
import pandera.pandas as pa
from chap_core.assessment.flat_representations import DIM_REGISTRY, DataDimension, FlatForecasts, FlatObserved, convert_backtest_observations_to_flat_observations, convert_backtest_to_flat_forecasts
import pandas as pd
import numpy as np

from chap_core.database.tables import BackTest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricSpec:
    group_by: tuple[DataDimension, ...] = ()
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

    def _make_schema(self) -> pa.DataFrameSchema:
        cols: dict[str, pa.Column] = {}
        for d in self.spec.group_by:
            dtype, chk = DIM_REGISTRY[d]
            cols[d.value] = pa.Column(dtype, chk) if chk else pa.Column(dtype)
        cols[self.spec.metric_name] = pa.Column(float)
        return pa.DataFrameSchema(cols, strict=True, coerce=True)

    def get_name(self) -> str:
        return self.spec.metric_name

    def gives_highest_resolution(self) -> bool:
        """
        Returns True if the metric gives one number per location/time_period/horizon_distance combination. 
        """
        return len(self.spec.group_by) == 3

    def is_full_aggregate(self) -> bool:
        """
        Returns True if the metric gives only one number for the whole dataset
        """
        return len(self.spec.group_by) == 0 

class RMSE(MetricBase):
    """
    Root Mean Squared Error metric.
    Groups by location to give RMSE per location across all time periods and horizons.
    """
    spec = MetricSpec(group_by=(DataDimension.location,))

    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
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



class DetailedRMSE(MetricBase):
    """
    Detailed Root Mean Squared Error metric.
    Does not group - gives one RMSE value per location/time_period/horizon_distance combination.
    This provides the highest resolution view of model performance.
    """
    spec = MetricSpec(group_by=(DataDimension.location, DataDimension.time_period, DataDimension.horizon_distance))
    
    def compute(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
        # Merge observations with forecasts on location and time_period
        merged = forecasts.merge(
            observations[['location', 'time_period', 'disease_cases']],
            on=['location', 'time_period'],
            how='inner'
        )
        
        # Calculate squared error for each forecast
        merged['squared_error'] = (merged['forecast'] - merged['disease_cases']) ** 2
        
        # Average across samples for each location/time_period/horizon combination
        detailed_mse = (
            merged.groupby(['location', 'time_period', 'horizon_distance'], as_index=False)['squared_error']
            .mean()
        )
        
        # Take square root to get RMSE
        detailed_mse['metric'] = detailed_mse['squared_error'] ** 0.5
        
        # Return only the required columns
        return detailed_mse[['location', 'time_period', 'horizon_distance', 'metric']]


class DetailedCRPS(MetricBase):
    """
    Detailed Continuous Ranked Probability Score (CRPS) metric.
    Does not group - gives one CRPS value per location/time_period/horizon_distance combination.
    CRPS measures both calibration and sharpness of probabilistic forecasts.
    """
    spec = MetricSpec(
        group_by=(DataDimension.location, DataDimension.time_period, DataDimension.horizon_distance),
        metric_id='detailed_crps',
        description='CRPS per location, time period and horizon'
    )
    
    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # Merge observations with forecasts on location and time_period
        merged = forecasts.merge(
            observations[['location', 'time_period', 'disease_cases']],
            on=['location', 'time_period'],
            how='inner'
        )
        
        # Group by location, time_period, and horizon_distance to compute CRPS
        results = []
        for (location, time_period, horizon), group in merged.groupby(
            ['location', 'time_period', 'horizon_distance']
        ):
            # Get all sample values for this combination
            sample_values = group['forecast'].values
            # Get the observation (should be the same for all samples)
            obs_value = group['disease_cases'].iloc[0]
            
            # Calculate CRPS using the formula from database.py
            # CRPS = E[|X - obs|] - 0.5 * E[|X - X'|]
            term1 = np.mean(np.abs(sample_values - obs_value))
            term2 = 0.5 * np.mean(np.abs(sample_values[:, None] - sample_values[None, :]))
            crps = float(term1 - term2)
            
            results.append({
                'location': location,
                'time_period': time_period,
                'horizon_distance': horizon,
                'metric': crps
            })
        
        return pd.DataFrame(results)


class CRPSPerLocation(MetricBase):
    """
    Continuous Ranked Probability Score (CRPS) metric aggregated by location.
    Groups by location to give average CRPS per location across all time periods and horizons.
    """
    spec = MetricSpec(
        group_by=(DataDimension.location,),
        metric_id='crps_per_location',
        description='Average CRPS per location'
    )
    
    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # First compute detailed CRPS
        detailed_crps_metric = DetailedCRPS()
        detailed_results = detailed_crps_metric.compute(observations, forecasts)
        
        # Aggregate by location
        location_crps = (
            detailed_results.groupby('location', as_index=False)['metric']
            .mean()
        )
        
        return location_crps


class CRPS(MetricBase):
    """
    Continuous Ranked Probability Score (CRPS) metric for the entire dataset.
    Gives one CRPS value across all locations, time periods and horizons.
    """
    spec = MetricSpec(
        group_by=(),
        metric_id='crps',
        description='Overall CRPS across entire dataset'
    )
    
    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # First compute CRPS per location
        crps_per_location_metric = CRPSPerLocation()
        location_results = crps_per_location_metric.compute(observations, forecasts)
        
        # Aggregate across all locations to get overall CRPS
        overall_crps = location_results['metric'].mean()
        
        return pd.DataFrame({'metric': [overall_crps]})


class DetailedCRPSNorm(MetricBase):
    """
    Detailed Normalized Continuous Ranked Probability Score (CRPS) metric.
    Does not group - gives one normalized CRPS value per location/time_period/horizon_distance combination.
    CRPS is normalized by the range of observed values to make it comparable across different scales.
    """
    spec = MetricSpec(
        group_by=(DataDimension.location, DataDimension.time_period, DataDimension.horizon_distance),
        metric_id='detailed_crps_norm',
        description='Normalized CRPS per location, time period and horizon'
    )
    
    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # First compute regular CRPS for each location/time_period/horizon combination
        detailed_crps_metric = DetailedCRPS()
        detailed_crps_results = detailed_crps_metric.compute(observations, forecasts)
        
        # Calculate normalization factor based on range of all observed values
        obs_values = observations['disease_cases'].values
        obs_min, obs_max = obs_values.min(), obs_values.max()
        obs_range = obs_max - obs_min
        
        # Avoid division by zero if all observations are the same
        if obs_range == 0:
            # If all observations are identical, normalized CRPS is just the regular CRPS
            detailed_crps_results['metric'] = detailed_crps_results['metric']
        else:
            # Normalize CRPS by the range of observations
            detailed_crps_results['metric'] = detailed_crps_results['metric'] / obs_range
        
        return detailed_crps_results


class CRPSNorm(MetricBase):
    """
    Normalized Continuous Ranked Probability Score (CRPS) metric aggregated by location.
    Groups by location to give average normalized CRPS per location across all time periods and horizons.
    """
    spec = MetricSpec(
        group_by=(DataDimension.location,),
        metric_id='crps_norm',
        description='Average normalized CRPS per location'
    )
    
    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # First compute detailed normalized CRPS
        detailed_crps_norm_metric = DetailedCRPSNorm()
        detailed_results = detailed_crps_norm_metric.compute(observations, forecasts)
        
        # Aggregate by location
        location_crps_norm = (
            detailed_results.groupby('location', as_index=False)['metric']
            .mean()
        )
        
        return location_crps_norm


class IsWithin10th90thDetailed(MetricBase):
    """
    Detailed metric checking if observation falls within 10th-90th percentile of forecast samples.
    Does not group - gives one binary value (0 or 1) per location/time_period/horizon_distance combination.
    Returns 1 if observation is within the 10th-90th percentile range, 0 otherwise.
    """
    spec = MetricSpec(
        group_by=(DataDimension.location, DataDimension.time_period, DataDimension.horizon_distance),
        metric_id='is_within_10th_90th_detailed',
        description='Binary indicator if observation is within 10th-90th percentile per location, time period and horizon'
    )
    
    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # Merge observations with forecasts on location and time_period
        merged = forecasts.merge(
            observations[['location', 'time_period', 'disease_cases']],
            on=['location', 'time_period'],
            how='inner'
        )
        
        # Group by location, time_period, and horizon_distance to compute percentile coverage
        results = []
        for (location, time_period, horizon), group in merged.groupby(
            ['location', 'time_period', 'horizon_distance']
        ):
            # Get all sample values for this combination
            sample_values = group['forecast'].values
            # Get the observation (should be the same for all samples)
            obs_value = group['disease_cases'].iloc[0]
            
            # Calculate 10th and 90th percentiles of the samples
            low, high = np.percentile(sample_values, [10, 90])
            # Check if observation falls within this range
            is_within_range = 1.0 if (low <= obs_value <= high) else 0.0
            
            results.append({
                'location': location,
                'time_period': time_period,
                'horizon_distance': horizon,
                'metric': is_within_range
            })
        
        return pd.DataFrame(results)


class IsWithin25th75thDetailed(MetricBase):
    """
    Detailed metric checking if observation falls within 25th-75th percentile of forecast samples.
    Does not group - gives one binary value (0 or 1) per location/time_period/horizon_distance combination.
    Returns 1 if observation is within the 25th-75th percentile range, 0 otherwise.
    """
    spec = MetricSpec(
        group_by=(DataDimension.location, DataDimension.time_period, DataDimension.horizon_distance),
        metric_id='is_within_25th_75th_detailed',
        description='Binary indicator if observation is within 25th-75th percentile per location, time period and horizon'
    )
    
    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # Merge observations with forecasts on location and time_period
        merged = forecasts.merge(
            observations[['location', 'time_period', 'disease_cases']],
            on=['location', 'time_period'],
            how='inner'
        )
        
        # Group by location, time_period, and horizon_distance to compute percentile coverage
        results = []
        for (location, time_period, horizon), group in merged.groupby(
            ['location', 'time_period', 'horizon_distance']
        ):
            # Get all sample values for this combination
            sample_values = group['forecast'].values
            # Get the observation (should be the same for all samples)
            obs_value = group['disease_cases'].iloc[0]
            
            # Calculate 25th and 75th percentiles of the samples
            low, high = np.percentile(sample_values, [25, 75])
            # Check if observation falls within this range
            is_within_range = 1.0 if (low <= obs_value <= high) else 0.0
            
            results.append({
                'location': location,
                'time_period': time_period,
                'horizon_distance': horizon,
                'metric': is_within_range
            })
        
        return pd.DataFrame(results)


class RatioWithin10th90thPerLocation(MetricBase):
    """
    Ratio of observations within 10th-90th percentile, aggregated by location.
    Groups by location to give the proportion of forecasts where observation fell within range.
    """
    spec = MetricSpec(
        group_by=(DataDimension.location,),
        metric_id='ratio_within_10th_90th_per_location',
        description='Ratio of observations within 10th-90th percentile per location'
    )
    
    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # First compute detailed metric
        detailed_metric = IsWithin10th90thDetailed()
        detailed_results = detailed_metric.compute(observations, forecasts)
        
        # Aggregate by location (mean of binary values gives ratio)
        location_ratios = (
            detailed_results.groupby('location', as_index=False)['metric']
            .mean()
        )
        
        return location_ratios


class RatioWithin10th90th(MetricBase):
    """
    Overall ratio of observations within 10th-90th percentile for entire dataset.
    Gives one ratio value across all locations, time periods and horizons.
    """
    spec = MetricSpec(
        group_by=(),
        metric_id='ratio_within_10th_90th',
        description='Overall ratio of observations within 10th-90th percentile'
    )
    
    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # First compute ratio per location
        ratio_per_location_metric = RatioWithin10th90thPerLocation()
        location_results = ratio_per_location_metric.compute(observations, forecasts)
        
        # Aggregate across all locations to get overall ratio
        overall_ratio = location_results['metric'].mean()
        
        return pd.DataFrame({'metric': [overall_ratio]})


class RatioWithin25th75thPerLocation(MetricBase):
    """
    Ratio of observations within 25th-75th percentile, aggregated by location.
    Groups by location to give the proportion of forecasts where observation fell within range.
    """
    spec = MetricSpec(
        group_by=(DataDimension.location,),
        metric_id='ratio_within_25th_75th_per_location',
        description='Ratio of observations within 25th-75th percentile per location'
    )
    
    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # First compute detailed metric
        detailed_metric = IsWithin25th75thDetailed()
        detailed_results = detailed_metric.compute(observations, forecasts)
        
        # Aggregate by location (mean of binary values gives ratio)
        location_ratios = (
            detailed_results.groupby('location', as_index=False)['metric']
            .mean()
        )
        
        return location_ratios


class RatioWithin25th75th(MetricBase):
    """
    Overall ratio of observations within 25th-75th percentile for entire dataset.
    Gives one ratio value across all locations, time periods and horizons.
    """
    spec = MetricSpec(
        group_by=(),
        metric_id='ratio_within_25th_75th',
        description='Overall ratio of observations within 25th-75th percentile'
    )
    
    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # First compute ratio per location
        ratio_per_location_metric = RatioWithin25th75thPerLocation()
        location_results = ratio_per_location_metric.compute(observations, forecasts)
        
        # Aggregate across all locations to get overall ratio
        overall_ratio = location_results['metric'].mean()
        
        return pd.DataFrame({'metric': [overall_ratio]})


class TestMetricDetailed(MetricBase):
    """
    Test metric that counts the number of forecast samples per location/time_period/horizon_distance.
    Useful for debugging and verifying data structure correctness.
    Returns the count of samples for each combination.
    """
    spec = MetricSpec(
        group_by=(DataDimension.location, DataDimension.time_period, DataDimension.horizon_distance),
        metric_id='test_sample_count_detailed',
        description='Number of forecast samples per location, time period and horizon'
    )
    
    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # Group by location, time_period, and horizon_distance to count samples
        sample_counts = (
            forecasts.groupby(['location', 'time_period', 'horizon_distance'], as_index=False)
            .size()
            .rename(columns={'size': 'metric'})
        )
        
        # Convert metric to float to match schema expectations
        sample_counts['metric'] = sample_counts['metric'].astype(float)
        
        return sample_counts


class TestMetric(MetricBase):
    """
    Test metric that counts the total number of forecast samples in the entire dataset.
    Returns a single number representing the total sample count.
    """
    spec = MetricSpec(
        group_by=(),
        metric_id='test_sample_count',
        description='Total number of forecast samples in dataset'
    )
    
    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # Count total number of rows in forecasts (each row is one sample)
        total_samples = float(len(forecasts))
        return pd.DataFrame({'metric': [total_samples]})


available_metrics: dict[str, MetricBase] = {
    'rmse': RMSE,
    'mae': MAE,
    'detailed_rmse': DetailedRMSE,
    'detailed_crps': DetailedCRPS,
    'crps_per_location': CRPSPerLocation,
    'crps': CRPS,
    'detailed_crps_norm': DetailedCRPSNorm,
    'crps_norm': CRPSNorm,
    'is_within_10th_90th_detailed': IsWithin10th90thDetailed,
    'is_within_25th_75th_detailed': IsWithin25th75thDetailed,
    'ratio_within_10th_90th_per_location': RatioWithin10th90thPerLocation,
    'ratio_within_10th_90th': RatioWithin10th90th,
    'ratio_within_25th_75th_per_location': RatioWithin25th75thPerLocation,
    'ratio_within_25th_75th': RatioWithin25th75th,
    'test_sample_count_detailed': TestMetricDetailed,
    'test_sample_count': TestMetric,
}



def compute_all_aggregated_metrics_from_backtest(backtest: BackTest) -> dict[str, float]:
    relevant_metrics = {id: metric for id, metric in available_metrics.items() if metric().is_full_aggregate()}
    logger.info(f"Relevant metrics for aggregation: {relevant_metrics.keys()}")

    # Convert to flat representation
    flat_forecasts = FlatForecasts(convert_backtest_to_flat_forecasts(backtest.forecasts))
    flat_observations = FlatObserved(convert_backtest_observations_to_flat_observations(backtest.dataset.observations))

    results = {}
    for id, metric_cls in relevant_metrics.items():
        metric = metric_cls()
        metric_df = metric.get_metric(flat_observations, flat_forecasts)
        if len(metric_df) != 1:
            raise ValueError(f"Metric {id} was expected to return a single aggregated value, but got {len(metric_df)} rows.")
        results[id] = float(metric_df[metric.spec.metric_name].iloc[0])

    return results
