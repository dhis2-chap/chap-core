"""
Evaluation abstraction for model evaluation results.

This module provides a database-agnostic interface for working with model
evaluation results, enabling better code reuse between REST API and CLI workflows.
"""

import datetime
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Iterable, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from chap_core.api_types import BackTestParams

from chap_core.data import DataSet as _DataSet
from chap_core.assessment.flat_representations import (
    FlatForecasts,
    FlatObserved,
    convert_backtest_observations_to_flat_observations,
    convert_backtest_to_flat_forecasts,
)
from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB
from chap_core.datatypes import SamplesWithTruth
from chap_core.rest_api.data_models import BackTestCreate
from chap_core.time_period import TimePeriod
from chap_core.database.tables import BackTest, BackTestForecast
from chap_core.database.dataset_tables import DataSet, Observation

try:
    from chap_core import __version__ as CHAP_VERSION
except ImportError:
    CHAP_VERSION = "unknown"


def _flat_data_to_xarray(flat_data: "FlatEvaluationData", model_metadata: dict) -> xr.Dataset:
    """
    Convert FlatEvaluationData to xarray.Dataset.

    Args:
        flat_data: FlatEvaluationData containing forecasts and observations
        model_metadata: Dictionary with model information (name, configuration, version)

    Returns:
        xarray.Dataset with dimensions (location, time_period, horizon_distance, sample)
    """
    forecasts_df = pd.DataFrame(flat_data.forecasts).copy()
    observations_df = pd.DataFrame(flat_data.observations).copy()

    # Set multi-index for forecasts: location, time_period, horizon_distance, sample
    forecasts_indexed = forecasts_df.set_index(["location", "time_period", "horizon_distance", "sample"])

    # Convert to xarray - this creates a DataArray with the index levels as dimensions
    forecast_da = forecasts_indexed["forecast"].to_xarray()

    # Set multi-index for observations: location, time_period
    observations_indexed = observations_df.set_index(["location", "time_period"])
    observed_da = observations_indexed["disease_cases"].to_xarray()

    # Combine into a single dataset
    data_vars = {
        "forecast": forecast_da,
        "observed": observed_da,
    }

    # Add historical observations if present
    if flat_data.historical_observations is not None:
        historical_df = pd.DataFrame(flat_data.historical_observations).copy()
        if not historical_df.empty:
            # Use different dimension name to avoid conflicts with test period observations
            historical_indexed = historical_df.set_index(["location", "time_period"])
            historical_indexed = historical_indexed.rename_axis(index={"time_period": "historical_time_period"})
            historical_da = historical_indexed["disease_cases"].to_xarray()
            data_vars["historical_observed"] = historical_da

    ds = xr.Dataset(data_vars)

    # Add global attributes
    ds.attrs.update(
        {
            "title": "CHAP Model Evaluation Results",
            "model_name": model_metadata.get("model_name", ""),
            "model_configuration": json.dumps(model_metadata.get("model_configuration", {})),
            "model_version": model_metadata.get("model_version", ""),
            "created_date": datetime.datetime.now().isoformat(),
            "split_periods": json.dumps(model_metadata.get("split_periods", [])),
            "org_units": json.dumps(model_metadata.get("org_units", [])),
            "historical_context_periods": model_metadata.get("historical_context_periods", 0),
            "chap_version": CHAP_VERSION,
        }
    )

    return ds


def _xarray_to_flat_data(ds: xr.Dataset) -> "FlatEvaluationData":
    """
    Convert xarray.Dataset back to FlatEvaluationData.

    Args:
        ds: xarray.Dataset with forecast and observed variables

    Returns:
        FlatEvaluationData with reconstructed forecasts and observations
    """
    # Convert forecast DataArray to DataFrame - pandas will handle multi-index automatically
    forecasts_df = ds["forecast"].to_dataframe().reset_index()

    # Convert observed DataArray to DataFrame
    # The DataArray name is 'observed' but we need it as 'disease_cases' for FlatObserved
    observations_df = ds["observed"].to_dataframe().reset_index()
    if "observed" in observations_df.columns:
        observations_df = observations_df.rename(columns={"observed": "disease_cases"})

    # Drop NaN forecasts (missing data)
    forecasts_df = forecasts_df.dropna(subset=["forecast"])

    # Drop NaN observations
    observations_df = observations_df.dropna(subset=["disease_cases"])

    # Convert horizon_distance and sample to int64 (NetCDF may have stored as int32)
    if "horizon_distance" in forecasts_df.columns:
        forecasts_df["horizon_distance"] = forecasts_df["horizon_distance"].astype(np.int64)
    if "sample" in forecasts_df.columns:
        forecasts_df["sample"] = forecasts_df["sample"].astype(np.int64)

    # Load historical observations if present (backwards compatible)
    historical_observations = None
    if "historical_observed" in ds:
        historical_df = ds["historical_observed"].to_dataframe().reset_index()
        if "historical_observed" in historical_df.columns:
            historical_df = historical_df.rename(columns={"historical_observed": "disease_cases"})
        # Rename dimension back to time_period for consistency
        if "historical_time_period" in historical_df.columns:
            historical_df = historical_df.rename(columns={"historical_time_period": "time_period"})
        historical_df = historical_df.dropna(subset=["disease_cases"])
        if not historical_df.empty:
            historical_observations = FlatObserved(historical_df)

    return FlatEvaluationData(
        forecasts=FlatForecasts(forecasts_df),
        observations=FlatObserved(observations_df),
        historical_observations=historical_observations,
    )


@dataclass
class FlatEvaluationData:
    """
    Container for flat representations of evaluation data.

    Combines forecasts and observations which are always used together
    for metric computation and visualization.

    Attributes:
        forecasts: Flat representation of forecast samples
        observations: Flat representation of observed values (test periods for evaluation)
        historical_observations: Flat representation of historical observed values
            before split periods (for plotting context). Optional for backwards compatibility.
    """

    forecasts: FlatForecasts
    observations: FlatObserved
    historical_observations: Optional[FlatObserved] = None


class EvaluationBase(ABC):
    """
    Abstract base class for evaluation results.

    An Evaluation represents the complete results of evaluating a model:
    - Forecasts (with samples/quantiles)
    - Observations (ground truth)
    - Metadata (locations, split periods)

    This abstraction is database-agnostic and can be implemented by
    different concrete classes (database-backed, in-memory, etc.).
    """

    @abstractmethod
    def to_flat(self) -> FlatEvaluationData:
        """
        Export evaluation data as flat representations.

        Returns:
            FlatEvaluationData containing FlatForecasts and FlatObserved objects
        """
        pass

    @abstractmethod
    def get_org_units(self) -> List[str]:
        """
        Get list of locations included in this evaluation.

        Returns:
            List of location identifiers (org_units)
        """
        pass

    @abstractmethod
    def get_split_periods(self) -> List[str]:
        """
        Get list of train/test split periods used in evaluation.

        Returns:
            List of period identifiers (e.g., ["2024-01", "2024-02"])
        """
        pass

    @classmethod
    @abstractmethod
    def from_backtest(cls, backtest: "BackTest") -> "EvaluationBase":
        """
        Create Evaluation from database BackTest object.

        All implementations must support loading from database.

        Args:
            backtest: Database BackTest object (with relationships loaded)

        Returns:
            Evaluation instance
        """
        pass

    @classmethod
    @abstractmethod
    def from_samples_with_truth(
        cls,
        evaluation_results: Iterable[_DataSet[SamplesWithTruth]],
        last_train_period: TimePeriod,
        configured_model: ConfiguredModelDB,
    ) -> "EvaluationBase": ...


class Evaluation(EvaluationBase):
    """
    Evaluation implementation backed by database BackTest model.

    This wraps an existing BackTest object and provides the
    EvaluationBase interface without modifying the database schema.
    """

    def __init__(
        self,
        backtest: "BackTest",
        historical_observations: Optional[List[Observation]] = None,
        historical_context_periods: int = 0,
    ):
        """
        Initialize Evaluation with a BackTest object.

        Args:
            backtest: Database BackTest object (with relationships loaded)
            historical_observations: Optional list of Observation objects for historical
                context (periods before split points, for plotting)
            historical_context_periods: Number of periods of historical context stored
        """
        self._backtest = backtest
        self._historical_observations = historical_observations or []
        self._historical_context_periods = historical_context_periods
        self._flat_data_cache: Optional[FlatEvaluationData] = None

    @classmethod
    def from_backtest(cls, backtest: "BackTest") -> "Evaluation":
        """
        Create Evaluation from database BackTest object.

        Args:
            backtest: Database BackTest object (with relationships loaded)

        Returns:
            Evaluation instance wrapping the BackTest
        """
        return cls(backtest)

    @classmethod
    def from_samples_with_truth(
        cls,
        evaluation_results: Iterable[_DataSet[SamplesWithTruth]],
        last_train_period: TimePeriod,
        configured_model: ConfiguredModelDB,
        info: BackTestCreate,
        historical_observations: Optional[List[Observation]] = None,
        historical_context_periods: int = 0,
    ):
        info.created = datetime.datetime.now()
        backtest = BackTest(
            **info.model_dump()
            | {"model_db_id": configured_model.id, "model_template_version": configured_model.model_template.version}
        )
        org_units = set([])
        split_points = set([])
        observations = []

        for eval_result in evaluation_results:
            first_period: TimePeriod = eval_result.period_range[0]
            split_points.add(first_period.id)
            for location, samples_with_truth in eval_result.items():
                # NOTE: samples_with_truth is class datatypes.SamplesWithTruth
                org_units.add(location)
                for period, sample_values, disease_cases in zip(
                    eval_result.period_range, samples_with_truth.samples, samples_with_truth.disease_cases
                ):
                    # add forecast series for this period
                    forecast = BackTestForecast(
                        period=period.id,
                        org_unit=location,
                        last_train_period=last_train_period.id,
                        last_seen_period=first_period.id,
                        values=sample_values.tolist(),
                    )
                    backtest.forecasts.append(forecast)

                    # add observation for this period/location
                    observation = Observation(
                        period=period.id,
                        org_unit=location,
                        value=float(disease_cases) if disease_cases is not None else None,
                        feature_name="disease_cases",
                    )
                    observations.append(observation)

        backtest.org_units = list(org_units)
        backtest.split_periods = list(split_points)

        # Deduplicate observations by (period, org_unit) - keep first occurrence
        seen = set()
        unique_observations = []
        for obs in observations:
            key = (obs.period, obs.org_unit)
            if key not in seen:
                seen.add(key)
                unique_observations.append(obs)

        # Create a minimal in-memory dataset with observations
        # Only create this if there's no existing dataset (CLI path)
        # For database path, the dataset relationship will be loaded via dataset_id
        # Check if dataset_id is 0 (CLI dummy value) to determine if we need an in-memory dataset
        if info.dataset_id == 0:
            backtest.dataset = DataSet(
                name="cli_evaluation_dataset",
                observations=unique_observations,
            )

        return cls(
            backtest,
            historical_observations=historical_observations,
            historical_context_periods=historical_context_periods,
        )

    @classmethod
    def create(
        cls,
        configured_model: ConfiguredModelDB,
        estimator,
        dataset: _DataSet,
        backtest_params: "BackTestParams",
        backtest_name: str = "evaluation",
        historical_context_years: int = 6,
    ) -> "Evaluation":
        """
        Create an Evaluation by running a backtest.

        Factory method that handles the complete backtest workflow:
        1. Run backtest with provided estimator
        2. Create Evaluation from results

        Args:
            configured_model: Configured model database object with metadata
            estimator: Model estimator instance ready for training/prediction
            dataset: Dataset to evaluate on
            backtest_params: Backtest execution parameters (n_periods, n_splits, stride)
            backtest_name: Name for the backtest (default: "evaluation")
            historical_context_years: Years of historical data to include for plotting
                context (default: 6). Number of periods is calculated based on dataset
                period type (e.g., 6 years = 312 weeks or 72 months).

        Returns:
            Evaluation instance with backtest results
        """
        from chap_core.assessment.dataset_splitting import train_test_generator
        from chap_core.assessment.prediction_evaluator import backtest

        # Run backtest
        evaluation_results = backtest(
            estimator=estimator,
            data=dataset,
            prediction_length=backtest_params.n_periods,
            n_test_sets=backtest_params.n_splits,
            stride=backtest_params.stride,
        )

        # Prepare metadata
        train, _ = train_test_generator(
            dataset, backtest_params.n_periods, backtest_params.n_splits, stride=backtest_params.stride
        )
        last_train_period = train.period_range[-1]

        backtest_info = BackTestCreate(
            name=backtest_name,
            dataset_id=0,
            model_id=configured_model.id,
        )

        # Calculate number of periods based on dataset period type
        historical_context_periods = cls._calculate_periods_from_years(
            dataset=dataset,
            years=historical_context_years,
        )

        # Extract historical observations from the dataset for plotting context
        historical_observations = cls._extract_historical_observations(
            dataset=dataset,
            up_to_period=last_train_period,
            n_periods=historical_context_periods,
        )

        # Create Evaluation
        return cls.from_samples_with_truth(
            evaluation_results=evaluation_results,
            last_train_period=last_train_period,
            configured_model=configured_model,
            info=backtest_info,
            historical_observations=historical_observations,
            historical_context_periods=historical_context_periods,
        )

    @classmethod
    def _calculate_periods_from_years(cls, dataset: _DataSet, years: int) -> int:
        """
        Calculate number of periods from years based on dataset period type.

        Args:
            dataset: Dataset to get period type from
            years: Number of years

        Returns:
            Number of periods (e.g., 6 years = 312 weeks or 72 months)
        """
        # Get time delta from the first location's data
        first_location = next(iter(dataset.keys()))
        time_periods = dataset[first_location].time_period
        time_delta = time_periods.delta

        # Calculate periods per year based on time delta
        rd = time_delta._relative_delta
        if rd.days == 7 or rd.weeks == 1:
            # Weekly data
            periods_per_year = 52
        elif rd.months == 1:
            # Monthly data
            periods_per_year = 12
        elif rd.days == 1:
            # Daily data
            periods_per_year = 365
        elif rd.years == 1:
            # Yearly data
            periods_per_year = 1
        else:
            # Default to weekly if unknown
            periods_per_year = 52

        return years * periods_per_year

    @classmethod
    def _extract_historical_observations(
        cls,
        dataset: _DataSet,
        up_to_period: TimePeriod,
        n_periods: int,
    ) -> List[Observation]:
        """
        Extract historical observations from a dataset for plotting context.

        Args:
            dataset: Dataset containing disease_cases time series
            up_to_period: Include observations up to and including this period
            n_periods: Maximum number of periods to include

        Returns:
            List of Observation objects for historical context
        """
        observations = []

        for location in dataset.keys():
            location_data = dataset[location]
            time_periods = location_data.time_period
            disease_cases = location_data.disease_cases

            # Find the index of up_to_period (inclusive)
            end_idx = None
            for i, period in enumerate(time_periods):
                if period.id <= up_to_period.id:
                    end_idx = i

            if end_idx is None:
                continue

            # Calculate start index based on n_periods
            start_idx = max(0, end_idx - n_periods + 1)

            # Extract observations for this location
            for i in range(start_idx, end_idx + 1):
                period = time_periods[i]
                value = disease_cases[i]

                # Skip NaN values
                if value is not None and not np.isnan(value):
                    observation = Observation(
                        period=period.id,
                        org_unit=location,
                        value=float(value),
                        feature_name="disease_cases",
                    )
                    observations.append(observation)

        return observations

    def to_backtest(self) -> "BackTest":
        """
        Get underlying database BackTest object.

        Returns:
            BackTest database model
        """
        return self._backtest

    def to_flat(self) -> FlatEvaluationData:
        """
        Export evaluation data using existing conversion functions.

        Results are cached for performance. Repeated calls return the cached result.

        Returns:
            FlatEvaluationData containing forecasts, observations, and historical observations
        """
        if self._flat_data_cache is None:
            forecasts_df = convert_backtest_to_flat_forecasts(self._backtest.forecasts)
            observations_df = convert_backtest_observations_to_flat_observations(self._backtest.dataset.observations)

            # Convert historical observations if present
            historical_observations = None
            if self._historical_observations:
                historical_df = convert_backtest_observations_to_flat_observations(self._historical_observations)
                if not historical_df.empty:
                    historical_observations = FlatObserved(historical_df)

            self._flat_data_cache = FlatEvaluationData(
                forecasts=FlatForecasts(forecasts_df),
                observations=FlatObserved(observations_df),
                historical_observations=historical_observations,
            )
        return self._flat_data_cache

    def get_org_units(self) -> List[str]:
        """
        Get locations from BackTest metadata.

        Returns:
            List of location identifiers
        """
        return self._backtest.org_units

    def get_split_periods(self) -> List[str]:
        """
        Get split periods from BackTest metadata.

        Returns:
            List of period identifiers
        """
        return self._backtest.split_periods

    def to_file(
        self,
        filepath: Union[str, Path],
        model_name: Optional[str] = None,
        model_configuration: Optional[dict] = None,
        model_version: Optional[str] = None,
    ) -> None:
        """
        Export evaluation to NetCDF file using xarray.

        Args:
            filepath: Path to output NetCDF file
            model_name: Name of the model (optional)
            model_configuration: Model configuration dictionary (optional)
            model_version: Model version string (optional)
        """
        flat_data = self.to_flat()

        model_metadata = {
            "model_name": model_name or "",
            "model_configuration": model_configuration or {},
            "model_version": model_version or "",
            "split_periods": self.get_split_periods(),
            "org_units": self.get_org_units(),
            "historical_context_periods": self._historical_context_periods,
        }

        ds = _flat_data_to_xarray(flat_data, model_metadata)
        ds.to_netcdf(filepath)

    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> "Evaluation":
        """
        Load evaluation from NetCDF file.

        Creates an in-memory BackTest object without database persistence.

        Args:
            filepath: Path to NetCDF file

        Returns:
            Evaluation instance
        """
        ds = xr.open_dataset(filepath)

        flat_data = _xarray_to_flat_data(ds)

        split_periods = json.loads(ds.attrs.get("split_periods", "[]"))
        org_units = json.loads(ds.attrs.get("org_units", "[]"))
        historical_context_periods = int(ds.attrs.get("historical_context_periods", 0))

        backtest = BackTest(
            name=f"Loaded from {Path(filepath).name}",
            org_units=org_units,
            split_periods=split_periods,
            forecasts=[],
            dataset_id=0,
        )

        forecasts_df = pd.DataFrame(flat_data.forecasts)

        # Group by (location, time_period, horizon_distance) to create BackTestForecast objects
        for (location, time_period, horizon_distance), group in forecasts_df.groupby(
            ["location", "time_period", "horizon_distance"]
        ):
            # Calculate last_seen_period from horizon_distance
            period = TimePeriod.parse(time_period)
            last_seen_period = period - (int(horizon_distance) * period.time_delta)

            # Get all sample values for this forecast, sorted by sample number
            values = group.sort_values("sample")["forecast"].tolist()

            forecast = BackTestForecast(
                period=time_period,
                org_unit=location,
                last_seen_period=last_seen_period.id,
                last_train_period=last_seen_period.id,
                values=values,
            )
            backtest.forecasts.append(forecast)

        # Create observations from flat data
        observations_df = pd.DataFrame(flat_data.observations)
        observations = []
        for _, row in observations_df.iterrows():
            observation = Observation(
                period=row["time_period"],
                org_unit=row["location"],
                value=float(row["disease_cases"]) if row["disease_cases"] is not None else None,
                feature_name="disease_cases",
            )
            observations.append(observation)

        # Create in-memory dataset with observations
        backtest.dataset = DataSet(
            name=f"dataset_from_{Path(filepath).name}",
            observations=observations,
        )

        # Load historical observations if present
        historical_observations = []
        if flat_data.historical_observations is not None:
            historical_df = pd.DataFrame(flat_data.historical_observations)
            for _, row in historical_df.iterrows():
                observation = Observation(
                    period=row["time_period"],
                    org_unit=row["location"],
                    value=float(row["disease_cases"]) if row["disease_cases"] is not None else None,
                    feature_name="disease_cases",
                )
                historical_observations.append(observation)

        ds.close()

        return cls(
            backtest,
            historical_observations=historical_observations,
            historical_context_periods=historical_context_periods,
        )
