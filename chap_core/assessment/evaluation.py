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
from typing import List, Optional, Iterable, Union

import numpy as np
import pandas as pd
import xarray as xr

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
    ds = xr.Dataset(
        {
            "forecast": forecast_da,
            "observed": observed_da,
        }
    )

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

    return FlatEvaluationData(
        forecasts=FlatForecasts(forecasts_df),
        observations=FlatObserved(observations_df),
    )


@dataclass
class FlatEvaluationData:
    """
    Container for flat representations of evaluation data.

    Combines forecasts and observations which are always used together
    for metric computation and visualization.

    Attributes:
        forecasts: Flat representation of forecast samples
        observations: Flat representation of observed values
    """

    forecasts: FlatForecasts
    observations: FlatObserved


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
        cls, evaluation_results: Iterable[_DataSet[SamplesWithTruth]], last_train_period: TimePeriod, model_id
    ) -> "EvaluationBase": ...


class Evaluation(EvaluationBase):
    """
    Evaluation implementation backed by database BackTest model.

    This wraps an existing BackTest object and provides the
    EvaluationBase interface without modifying the database schema.
    """

    def __init__(self, backtest: "BackTest"):
        """
        Initialize Evaluation with a BackTest object.

        Args:
            backtest: Database BackTest object (with relationships loaded)
        """
        self._backtest = backtest
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
    ):
        info.created = datetime.datetime.now()
        # org_units = list({location for ds in evaluation_results for location in ds.locations()})
        # split_points = list({er.period_range[0] for er in evaluation_results})
        backtest = BackTest(
            **info.dict()
            | {"model_db_id": configured_model.id, "model_template_version": configured_model.model_template.version}
        )
        org_units = set([])
        split_points = set([])
        # define metrics (for each period)
        evaluation_results = list(
            evaluation_results
        )  # hacky, to avoid metric funcs using up the iterable before we can loop all splitpoints
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

        backtest.org_units = list(org_units)
        backtest.split_periods = list(split_points)
        return cls.from_backtest(backtest)

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
            FlatEvaluationData containing forecasts and observations
        """
        if self._flat_data_cache is None:
            forecasts_df = convert_backtest_to_flat_forecasts(self._backtest.forecasts)
            observations_df = convert_backtest_observations_to_flat_observations(self._backtest.dataset.observations)

            self._flat_data_cache = FlatEvaluationData(
                forecasts=FlatForecasts(forecasts_df),
                observations=FlatObserved(observations_df),
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

        backtest = BackTest(
            name=f"Loaded from {Path(filepath).name}",
            org_units=org_units,
            split_periods=split_periods,
            forecasts=[],
        )

        forecasts_df = pd.DataFrame(flat_data.forecasts)
        for _, row in forecasts_df.iterrows():
            forecast = BackTestForecast(
                period=row["time_period"],
                org_unit=row["location"],
                last_seen_period=row["time_period"],
                last_train_period=row["time_period"],
                values=[],
            )

            same_location_period = forecasts_df[
                (forecasts_df["location"] == row["location"])
                & (forecasts_df["time_period"] == row["time_period"])
                & (forecasts_df["horizon_distance"] == row["horizon_distance"])
            ]
            forecast.values = same_location_period.sort_values("sample")["forecast"].tolist()

            if not any(
                f.period == forecast.period
                and f.org_unit == forecast.org_unit
                and f.last_seen_period == forecast.last_seen_period
                for f in backtest.forecasts
            ):
                backtest.forecasts.append(forecast)

        ds.close()

        return cls.from_backtest(backtest)
