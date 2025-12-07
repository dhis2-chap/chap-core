"""Tests for Evaluation serialization to/from NetCDF using xarray."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from chap_core.assessment.evaluation import Evaluation, FlatEvaluationData
from chap_core.assessment.flat_representations import FlatForecasts, FlatObserved


class TestEvaluationSerialization:
    """Tests for to_file() and from_file() methods."""

    def test_to_file_creates_netcdf(self, backtest, tmp_path):
        """Test that to_file creates a NetCDF file."""
        evaluation = Evaluation.from_backtest(backtest)
        output_file = tmp_path / "test_evaluation.nc"

        evaluation.to_file(
            filepath=output_file,
            model_name="TestModel",
            model_configuration={"param1": "value1"},
            model_version="1.0.0",
        )

        assert output_file.exists()
        assert output_file.suffix == ".nc"

    def test_to_file_creates_valid_netcdf_structure(self, backtest, tmp_path):
        """Test that to_file creates a NetCDF with correct structure."""
        evaluation = Evaluation.from_backtest(backtest)
        output_file = tmp_path / "test_evaluation.nc"

        evaluation.to_file(filepath=output_file, model_name="TestModel")

        ds = xr.open_dataset(output_file)

        assert "forecast" in ds.data_vars
        assert "observed" in ds.data_vars

        assert "location" in ds.coords
        assert "time_period" in ds.coords
        assert "horizon_distance" in ds.coords
        assert "sample" in ds.coords

        ds.close()

    def test_to_file_includes_metadata(self, backtest, tmp_path):
        """Test that to_file includes model metadata in global attributes."""
        evaluation = Evaluation.from_backtest(backtest)
        output_file = tmp_path / "test_evaluation.nc"

        model_config = {"learning_rate": 0.001, "epochs": 100}

        evaluation.to_file(
            filepath=output_file,
            model_name="TestModel",
            model_configuration=model_config,
            model_version="2.1.0",
        )

        ds = xr.open_dataset(output_file)

        assert ds.attrs["title"] == "CHAP Model Evaluation Results"
        assert ds.attrs["model_name"] == "TestModel"
        assert json.loads(ds.attrs["model_configuration"]) == model_config
        assert ds.attrs["model_version"] == "2.1.0"
        assert "created_date" in ds.attrs
        assert "chap_version" in ds.attrs

        ds.close()

    def test_to_file_includes_split_periods_and_org_units(self, backtest, tmp_path):
        """Test that to_file includes split_periods and org_units in metadata."""
        backtest.org_units = ["OrgUnit1", "OrgUnit2"]
        backtest.split_periods = ["2022-01", "2022-02"]

        evaluation = Evaluation.from_backtest(backtest)
        output_file = tmp_path / "test_evaluation.nc"

        evaluation.to_file(filepath=output_file)

        ds = xr.open_dataset(output_file)

        split_periods = json.loads(ds.attrs["split_periods"])
        org_units = json.loads(ds.attrs["org_units"])

        assert split_periods == ["2022-01", "2022-02"]
        assert org_units == ["OrgUnit1", "OrgUnit2"]

        ds.close()

    def test_from_file_loads_evaluation(self, backtest, tmp_path):
        """Test that from_file loads an Evaluation instance."""
        evaluation = Evaluation.from_backtest(backtest)
        output_file = tmp_path / "test_evaluation.nc"
        evaluation.to_file(filepath=output_file)

        loaded_evaluation = Evaluation.from_file(output_file)

        assert isinstance(loaded_evaluation, Evaluation)

    def test_from_file_preserves_flat_data_structure(self, backtest, tmp_path):
        """Test that from_file preserves the flat data structure."""
        evaluation = Evaluation.from_backtest(backtest)
        output_file = tmp_path / "test_evaluation.nc"
        evaluation.to_file(filepath=output_file)

        loaded_evaluation = Evaluation.from_file(output_file)
        flat_data = loaded_evaluation.to_flat()

        assert isinstance(flat_data, FlatEvaluationData)
        assert isinstance(flat_data.forecasts, pd.DataFrame)
        assert isinstance(flat_data.observations, pd.DataFrame)

        # Check that the DataFrames have the expected columns
        assert set(flat_data.forecasts.columns) == {"location", "time_period", "horizon_distance", "sample", "forecast"}
        assert set(flat_data.observations.columns) == {"location", "time_period", "disease_cases"}

    def test_roundtrip_preserves_forecast_data(self, backtest, tmp_path):
        """Test that save/load preserves forecast data integrity."""
        evaluation = Evaluation.from_backtest(backtest)
        output_file = tmp_path / "test_evaluation.nc"

        original_flat = evaluation.to_flat()
        original_forecasts = (
            pd.DataFrame(original_flat.forecasts)
            .sort_values(["location", "time_period", "horizon_distance", "sample"])
            .reset_index(drop=True)
        )

        evaluation.to_file(filepath=output_file)
        loaded_evaluation = Evaluation.from_file(output_file)

        loaded_flat = loaded_evaluation.to_flat()
        loaded_forecasts = (
            pd.DataFrame(loaded_flat.forecasts)
            .sort_values(["location", "time_period", "horizon_distance", "sample"])
            .reset_index(drop=True)
        )

        assert len(original_forecasts) == len(loaded_forecasts)
        assert list(original_forecasts.columns) == list(loaded_forecasts.columns)

        for col in ["location", "time_period", "horizon_distance", "sample"]:
            assert original_forecasts[col].equals(loaded_forecasts[col])

        np.testing.assert_allclose(
            original_forecasts["forecast"].values,
            loaded_forecasts["forecast"].values,
            rtol=1e-6,
        )

    def test_roundtrip_preserves_observation_data(self, backtest, tmp_path):
        """Test that save/load preserves observation data integrity."""
        evaluation = Evaluation.from_backtest(backtest)
        output_file = tmp_path / "test_evaluation.nc"

        original_flat = evaluation.to_flat()
        original_obs = (
            pd.DataFrame(original_flat.observations).sort_values(["location", "time_period"]).reset_index(drop=True)
        )

        evaluation.to_file(filepath=output_file)
        loaded_evaluation = Evaluation.from_file(output_file)

        loaded_flat = loaded_evaluation.to_flat()
        loaded_obs = (
            pd.DataFrame(loaded_flat.observations).sort_values(["location", "time_period"]).reset_index(drop=True)
        )

        assert len(original_obs) == len(loaded_obs)
        assert list(original_obs.columns) == list(loaded_obs.columns)

        assert original_obs["location"].equals(loaded_obs["location"])
        assert original_obs["time_period"].equals(loaded_obs["time_period"])

        np.testing.assert_allclose(
            original_obs["disease_cases"].values,
            loaded_obs["disease_cases"].values,
            rtol=1e-6,
        )

    def test_to_file_with_string_path(self, backtest, tmp_path):
        """Test that to_file works with string path."""
        evaluation = Evaluation.from_backtest(backtest)
        output_file = str(tmp_path / "test_evaluation.nc")

        evaluation.to_file(filepath=output_file)

        assert Path(output_file).exists()

    def test_from_file_with_string_path(self, backtest, tmp_path):
        """Test that from_file works with string path."""
        evaluation = Evaluation.from_backtest(backtest)
        output_file = str(tmp_path / "test_evaluation.nc")
        evaluation.to_file(filepath=output_file)

        loaded_evaluation = Evaluation.from_file(output_file)

        assert isinstance(loaded_evaluation, Evaluation)

    def test_to_file_with_default_metadata(self, backtest, tmp_path):
        """Test that to_file works with default/empty metadata."""
        evaluation = Evaluation.from_backtest(backtest)
        output_file = tmp_path / "test_evaluation.nc"

        evaluation.to_file(filepath=output_file)

        ds = xr.open_dataset(output_file)

        assert ds.attrs["model_name"] == ""
        assert json.loads(ds.attrs["model_configuration"]) == {}
        assert ds.attrs["model_version"] == ""

        ds.close()

    def test_roundtrip_with_weekly_data(self, backtest_weeks, tmp_path):
        """Test roundtrip with weekly period data."""
        evaluation = Evaluation.from_backtest(backtest_weeks)
        output_file = tmp_path / "test_weekly.nc"

        original_flat = evaluation.to_flat()

        evaluation.to_file(filepath=output_file)
        loaded_evaluation = Evaluation.from_file(output_file)

        loaded_flat = loaded_evaluation.to_flat()

        assert len(original_flat.forecasts) == len(loaded_flat.forecasts)
        assert len(original_flat.observations) == len(loaded_flat.observations)

    @pytest.mark.skip(reason="sloooow")
    def test_to_file_handles_large_dataset(self, backtest_weeks_large, tmp_path):
        """Test that to_file can handle large datasets efficiently."""
        evaluation = Evaluation.from_backtest(backtest_weeks_large)
        output_file = tmp_path / "test_large.nc"

        evaluation.to_file(filepath=output_file, model_name="LargeModel")

        assert output_file.exists()

        ds = xr.open_dataset(output_file)
        assert "forecast" in ds.data_vars
        assert ds.attrs["model_name"] == "LargeModel"
        ds.close()

    @pytest.mark.skip(reason="sloooow")
    def test_roundtrip_large_dataset_preserves_data(self, backtest_weeks_large, tmp_path):
        """Test that roundtrip preserves data integrity for large datasets."""
        evaluation = Evaluation.from_backtest(backtest_weeks_large)
        output_file = tmp_path / "test_large_roundtrip.nc"

        original_flat = evaluation.to_flat()
        original_forecast_count = len(pd.DataFrame(original_flat.forecasts))

        evaluation.to_file(filepath=output_file)
        loaded_evaluation = Evaluation.from_file(output_file)

        loaded_flat = loaded_evaluation.to_flat()
        loaded_forecast_count = len(pd.DataFrame(loaded_flat.forecasts))

        assert original_forecast_count == loaded_forecast_count
        assert original_forecast_count > 10000


class TestXarrayHelperFunctions:
    """Tests for _flat_data_to_xarray and _xarray_to_flat_data helper functions."""

    def test_flat_data_to_xarray_creates_dataset(self, backtest):
        """Test that _flat_data_to_xarray creates a valid xarray.Dataset."""
        from chap_core.assessment.evaluation import _flat_data_to_xarray

        evaluation = Evaluation.from_backtest(backtest)
        flat_data = evaluation.to_flat()

        model_metadata = {
            "model_name": "TestModel",
            "model_configuration": {},
            "model_version": "1.0",
            "split_periods": [],
            "org_units": [],
        }

        ds = _flat_data_to_xarray(flat_data, model_metadata)

        assert isinstance(ds, xr.Dataset)
        assert "forecast" in ds.data_vars
        assert "observed" in ds.data_vars
        assert "horizon_distance" in ds.coords

    def test_xarray_to_flat_data_creates_flat_data(self, backtest, tmp_path):
        """Test that _xarray_to_flat_data creates valid FlatEvaluationData."""
        from chap_core.assessment.evaluation import _flat_data_to_xarray, _xarray_to_flat_data

        evaluation = Evaluation.from_backtest(backtest)
        original_flat = evaluation.to_flat()

        model_metadata = {
            "model_name": "TestModel",
            "model_configuration": {},
            "model_version": "1.0",
            "split_periods": [],
            "org_units": [],
        }

        ds = _flat_data_to_xarray(original_flat, model_metadata)
        reconstructed_flat = _xarray_to_flat_data(ds)

        assert isinstance(reconstructed_flat, FlatEvaluationData)
        assert isinstance(reconstructed_flat.forecasts, pd.DataFrame)
        assert isinstance(reconstructed_flat.observations, pd.DataFrame)

        # Check that the DataFrames have the expected columns
        assert set(reconstructed_flat.forecasts.columns) == {
            "location",
            "time_period",
            "horizon_distance",
            "sample",
            "forecast",
        }
        assert set(reconstructed_flat.observations.columns) == {"location", "time_period", "disease_cases"}
