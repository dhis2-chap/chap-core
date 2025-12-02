import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

from chap_core.models.external_model import ExternalModel
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.runners.mlflow_runner import MlFlowTrainPredictRunner


@pytest.fixture
def sample_dataset():
    """Create a sample dataset spanning 2019-2022 including COVID period."""
    # Create a DataFrame with monthly data from 2019-2022
    rows = []
    for year in range(2019, 2023):
        for month in range(1, 13):
            rows.append({
                "location": "location1",
                "time_period": f"{year}-{month:02d}",
                "disease_cases": 100.0,
                "rainfall": 50.0,
                "mean_temperature": 25.0
            })

    df = pd.DataFrame(rows)
    return DataSet.from_pandas(df)


@pytest.fixture
def mock_runner(tmp_path):
    """Create a mock MLFlow runner that doesn't actually run anything."""

    class MockRunner(MlFlowTrainPredictRunner):
        def train(self, *args, **kwargs):
            pass

        def predict(self, *args, **kwargs):
            pass

    return MockRunner(str(tmp_path / "model"))


def test_apply_chap_transformations_without_covid_mask(sample_dataset, mock_runner):
    """Test that data is unchanged when covid_mask is False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"user_option_values": {"chap__covid_mask": False}}

        model = ExternalModel(
            runner=mock_runner, working_dir=tmpdir, configuration=config
        )

        transformed_data = model._apply_chap_transformations(sample_dataset)

        # Data should be unchanged
        location_data = transformed_data["location1"]
        assert np.all(location_data.disease_cases == 100)
        assert not np.any(np.isnan(location_data.disease_cases))


def test_apply_chap_transformations_with_covid_mask(sample_dataset, mock_runner):
    """Test that COVID period data is masked when covid_mask is True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"user_option_values": {"chap__covid_mask": True}}

        model = ExternalModel(
            runner=mock_runner, working_dir=tmpdir, configuration=config
        )

        transformed_data = model._apply_chap_transformations(sample_dataset)

        # Check that COVID period has NaN values
        location_data = transformed_data["location1"]
        time_periods = location_data.time_period
        disease_cases = location_data.disease_cases

        # Check pre-COVID period (2019) - should be intact
        pre_covid_mask = np.array([str(p).startswith("2019") for p in time_periods])
        assert np.all(disease_cases[pre_covid_mask] == 100)

        # Check COVID period (2020-03 to 2021-12-31) - should be NaN
        covid_mask = np.array(
            [
                str(p) >= "2020-03" and str(p) <= "2021-12"
                for p in time_periods
            ]
        )
        assert np.all(np.isnan(disease_cases[covid_mask]))

        # Check post-COVID period (2022) - should be intact
        post_covid_mask = np.array([str(p).startswith("2022") for p in time_periods])
        assert np.all(disease_cases[post_covid_mask] == 100)


def test_apply_chap_transformations_with_empty_config(sample_dataset, mock_runner):
    """Test that data is unchanged with empty configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {}

        model = ExternalModel(
            runner=mock_runner, working_dir=tmpdir, configuration=config
        )

        transformed_data = model._apply_chap_transformations(sample_dataset)

        # Data should be unchanged (default is False)
        location_data = transformed_data["location1"]
        assert np.all(location_data.disease_cases == 100)
        assert not np.any(np.isnan(location_data.disease_cases))


def test_apply_chap_transformations_with_dict_configuration(sample_dataset, mock_runner):
    """Test that transformation works when configuration is a dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Configuration as dict (not ModelConfiguration object)
        config = {
            "user_option_values": {"chap__covid_mask": True, "other_param": 10}
        }

        model = ExternalModel(
            runner=mock_runner, working_dir=tmpdir, configuration=config
        )

        transformed_data = model._apply_chap_transformations(sample_dataset)

        # COVID period should be masked
        location_data = transformed_data["location1"]
        time_periods = location_data.time_period

        covid_mask = np.array(
            [
                str(p) >= "2020-03" and str(p) <= "2021-12"
                for p in time_periods
            ]
        )
        assert np.all(np.isnan(location_data.disease_cases[covid_mask]))


def test_apply_chap_transformations_preserves_other_fields(sample_dataset, mock_runner):
    """Test that COVID mask only affects disease_cases, not other fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"user_option_values": {"chap__covid_mask": True}}

        model = ExternalModel(
            runner=mock_runner, working_dir=tmpdir, configuration=config
        )

        transformed_data = model._apply_chap_transformations(sample_dataset)

        location_data = transformed_data["location1"]

        # Other fields should be unchanged
        assert np.all(location_data.rainfall == 50)
        assert np.all(location_data.mean_temperature == 25)
        assert not np.any(np.isnan(location_data.rainfall))
        assert not np.any(np.isnan(location_data.mean_temperature))
