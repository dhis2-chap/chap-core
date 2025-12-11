"""
Tests for EWARS model configuration through the configure-model endpoint.

These tests verify that various EWARS model configurations are correctly:
1. Created and stored in the database
2. Passed to the model during training
3. Written to the model_config.yaml file

Related to bug CLIM-282: Custom CHAP-EWARS model variants fail randomly with INLA resource issues.

Note: The ewars_template has:
- Required covariates: population
- User options: n_lags (integer), precision (number)
- Additional covariates allowed: rainfall, mean_temperature, etc.
"""

import logging
from unittest.mock import patch, MagicMock

import pytest
import yaml
from fastapi.testclient import TestClient

from chap_core.rest_api.v1.rest_api import app
from chap_core.rest_api.v1.routers.crud import ModelConfigurationCreate

logger = logging.getLogger(__name__)
client = TestClient(app)


def get_content(url):
    response = client.get(url)
    assert response.status_code == 200, response.json()
    return response.json()


def get_ewars_template_id():
    """Get the ID of the ewars_template model template."""
    url = "/v1/crud/model-templates"
    content = get_content(url)
    model = next(m for m in content if m["name"] == "ewars_template")
    return model["id"]


class TestEwarsModelConfigurationCreation:
    """Tests for creating EWARS model configurations through the API."""

    def test_create_ewars_config_minimal(self, celery_session_worker, dependency_overrides):
        """Test creating EWARS config with minimal parameters (only required)."""
        template_id = get_ewars_template_id()

        config = ModelConfigurationCreate(
            name="test_ewars_minimal",
            model_template_id=template_id,
            additional_continuous_covariates=[],
            user_option_values={"precision": 0.01, "n_lags": 3},
        )

        response = client.post("/v1/crud/configured-models", json=config.model_dump())
        assert response.status_code == 200, response.json()
        configured_model = response.json()
        assert "test_ewars_minimal" in configured_model["name"]
        assert configured_model["userOptionValues"]["precision"] == 0.01
        assert configured_model["userOptionValues"]["n_lags"] == 3

    def test_create_ewars_config_with_rainfall(self, celery_session_worker, dependency_overrides):
        """Test creating EWARS config with rainfall as additional covariate."""
        template_id = get_ewars_template_id()

        config = ModelConfigurationCreate(
            name="test_ewars_rainfall",
            model_template_id=template_id,
            additional_continuous_covariates=["rainfall"],
            user_option_values={"precision": 0.01, "n_lags": 3},
        )

        response = client.post("/v1/crud/configured-models", json=config.model_dump())
        assert response.status_code == 200, response.json()
        configured_model = response.json()
        assert configured_model["additionalContinuousCovariates"] == ["rainfall"]

    def test_create_ewars_config_with_weather_covariates(self, celery_session_worker, dependency_overrides):
        """Test creating EWARS config with rainfall and mean_temperature covariates."""
        template_id = get_ewars_template_id()

        config = ModelConfigurationCreate(
            name="test_ewars_weather_covariates",
            model_template_id=template_id,
            additional_continuous_covariates=["rainfall", "mean_temperature"],
            user_option_values={"precision": 0.01, "n_lags": 3},
        )

        response = client.post("/v1/crud/configured-models", json=config.model_dump())
        assert response.status_code == 200, response.json()
        configured_model = response.json()
        assert set(configured_model["additionalContinuousCovariates"]) == {
            "rainfall",
            "mean_temperature",
        }

    def test_create_ewars_config_with_covid_mask_false(self, celery_session_worker, dependency_overrides):
        """Test creating EWARS config with chap__covid_mask=false."""
        template_id = get_ewars_template_id()

        config = ModelConfigurationCreate(
            name="test_ewars_covid_mask_false",
            model_template_id=template_id,
            additional_continuous_covariates=["rainfall", "mean_temperature"],
            user_option_values={
                "precision": 0.01,
                "n_lags": 3,
                "chap__covid_mask": False,
            },
        )

        response = client.post("/v1/crud/configured-models", json=config.model_dump())
        assert response.status_code == 200, response.json()
        configured_model = response.json()
        assert configured_model["userOptionValues"]["chap__covid_mask"] is False

    def test_create_ewars_config_with_covid_mask_true(self, celery_session_worker, dependency_overrides):
        """Test creating EWARS config with chap__covid_mask=true."""
        template_id = get_ewars_template_id()

        config = ModelConfigurationCreate(
            name="test_ewars_covid_mask_true",
            model_template_id=template_id,
            additional_continuous_covariates=["rainfall", "mean_temperature"],
            user_option_values={
                "precision": 0.01,
                "n_lags": 3,
                "chap__covid_mask": True,
            },
        )

        response = client.post("/v1/crud/configured-models", json=config.model_dump())
        assert response.status_code == 200, response.json()
        configured_model = response.json()
        assert configured_model["userOptionValues"]["chap__covid_mask"] is True

    def test_create_ewars_config_bugfix_scenario(self, celery_session_worker, dependency_overrides):
        """
        Test creating a configuration similar to bug report CLIM-282.

        The bug report mentioned a barebone variant with:
        - n_lags=3
        - precision=0.01
        - covariates: rainfall, mean_temperature (in addition to required population)
        """
        template_id = get_ewars_template_id()

        config = ModelConfigurationCreate(
            name="test_ewars_bugfix_scenario",
            model_template_id=template_id,
            additional_continuous_covariates=["rainfall", "mean_temperature"],
            user_option_values={
                "n_lags": 3,
                "precision": 0.01,
                "chap__covid_mask": False,
            },
        )

        response = client.post("/v1/crud/configured-models", json=config.model_dump())
        assert response.status_code == 200, response.json()
        configured_model = response.json()

        # Verify all parameters match bug report scenario
        assert configured_model["userOptionValues"]["n_lags"] == 3
        assert configured_model["userOptionValues"]["precision"] == 0.01
        assert configured_model["userOptionValues"]["chap__covid_mask"] is False
        assert set(configured_model["additionalContinuousCovariates"]) == {
            "rainfall",
            "mean_temperature",
        }

    def test_create_ewars_config_different_n_lags(self, celery_session_worker, dependency_overrides):
        """Test creating EWARS configs with different n_lags values."""
        template_id = get_ewars_template_id()

        for n_lags in [1, 3, 5, 10]:
            config = ModelConfigurationCreate(
                name=f"test_ewars_n_lags_{n_lags}",
                model_template_id=template_id,
                additional_continuous_covariates=["rainfall"],
                user_option_values={"precision": 0.01, "n_lags": n_lags},
            )

            response = client.post("/v1/crud/configured-models", json=config.model_dump())
            assert response.status_code == 200, response.json()
            configured_model = response.json()
            assert configured_model["userOptionValues"]["n_lags"] == n_lags

    def test_create_ewars_config_different_precision(self, celery_session_worker, dependency_overrides):
        """Test creating EWARS configs with different precision values."""
        template_id = get_ewars_template_id()

        for idx, precision in enumerate([0.001, 0.01, 0.1, 1.0]):
            config = ModelConfigurationCreate(
                name=f"test_ewars_precision_{idx}",
                model_template_id=template_id,
                additional_continuous_covariates=["rainfall"],
                user_option_values={"precision": precision, "n_lags": 3},
            )

            response = client.post("/v1/crud/configured-models", json=config.model_dump())
            assert response.status_code == 200, response.json()
            configured_model = response.json()
            assert configured_model["userOptionValues"]["precision"] == precision


class TestEwarsModelConfigurationWrittenToFile:
    """Tests that verify the configuration is correctly written when training starts.

    These are unit tests that don't require celery/redis infrastructure.
    """

    def test_config_written_to_yaml_during_train(self):
        """Test that model configuration is correctly written to YAML during training."""
        from chap_core.models.external_model import ExternalModel
        from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB

        # Create a mock configuration that mimics what would be passed
        mock_config = MagicMock(spec=ConfiguredModelDB)
        mock_config.user_option_values = {
            "n_lags": 3,
            "precision": 0.01,
        }
        mock_config.additional_continuous_covariates = ["rainfall", "mean_temperature"]

        # Capture what gets written to YAML
        written_config = {}

        def mock_yaml_dump(data, file):
            nonlocal written_config
            written_config = data

        with patch.object(yaml, "dump", side_effect=mock_yaml_dump):
            # Create an ExternalModel instance
            mock_runner = MagicMock()
            model = ExternalModel(
                runner=mock_runner,
                name="test_model",
                configuration=mock_config,
            )

            # Create mock training data
            mock_train_data = MagicMock()
            mock_train_data.polygons = None
            mock_train_data.period_range = [MagicMock()]
            mock_train_data.period_range[0].__class__.__name__ = "Month"
            mock_train_data.to_pandas.return_value = MagicMock()
            mock_train_data.to_pandas.return_value.columns = MagicMock()
            mock_train_data.to_pandas.return_value.columns.tolist.return_value = []
            mock_train_data.to_pandas.return_value.to_csv = MagicMock()

            # Train should write config
            try:
                model.train(mock_train_data)
            except Exception:
                pass  # We expect it to fail since runner is mocked

        # Verify the config was captured
        assert written_config == mock_config

    def test_config_dict_format_during_train(self):
        """Test that configuration dict format is correct when passed to model."""
        from chap_core.models.external_model import ExternalModel

        # Test with dict configuration
        config_dict = {
            "user_option_values": {
                "n_lags": 3,
                "precision": 0.01,
                "chap__covid_mask": False,
            },
            "additional_continuous_covariates": ["rainfall", "mean_temperature"],
        }

        written_config = {}

        def mock_yaml_dump(data, file):
            nonlocal written_config
            written_config = data

        with patch.object(yaml, "dump", side_effect=mock_yaml_dump):
            mock_runner = MagicMock()
            model = ExternalModel(
                runner=mock_runner,
                name="test_model",
                configuration=config_dict,
            )

            mock_train_data = MagicMock()
            mock_train_data.polygons = None
            mock_train_data.period_range = [MagicMock()]
            mock_train_data.period_range[0].__class__.__name__ = "Month"
            mock_train_data.to_pandas.return_value = MagicMock()
            mock_train_data.to_pandas.return_value.columns = MagicMock()
            mock_train_data.to_pandas.return_value.columns.tolist.return_value = []
            mock_train_data.to_pandas.return_value.to_csv = MagicMock()

            try:
                model.train(mock_train_data)
            except Exception:
                pass

        # Verify the config dict matches
        assert written_config == config_dict
        assert written_config["user_option_values"]["n_lags"] == 3
        assert written_config["user_option_values"]["precision"] == 0.01
        assert written_config["user_option_values"]["chap__covid_mask"] is False
        assert "rainfall" in written_config["additional_continuous_covariates"]
        assert "mean_temperature" in written_config["additional_continuous_covariates"]


class TestEwarsConfigurationValidation:
    """Tests for validation of EWARS model configurations."""

    def test_invalid_precision_type(self, celery_session_worker, dependency_overrides):
        """Test that invalid precision type is rejected."""
        template_id = get_ewars_template_id()

        config = ModelConfigurationCreate(
            name="test_ewars_invalid_precision",
            model_template_id=template_id,
            additional_continuous_covariates=[],
            user_option_values={"precision": "invalid", "n_lags": 3},
        )

        # The API raises ValueError for invalid types, which becomes a 500 error
        with pytest.raises(ValueError, match="Invalid user options"):
            client.post("/v1/crud/configured-models", json=config.model_dump())

    def test_invalid_n_lags_type(self, celery_session_worker, dependency_overrides):
        """Test that invalid n_lags type is rejected."""
        template_id = get_ewars_template_id()

        config = ModelConfigurationCreate(
            name="test_ewars_invalid_n_lags",
            model_template_id=template_id,
            additional_continuous_covariates=[],
            user_option_values={"precision": 0.01, "n_lags": "invalid"},
        )

        # The API raises ValueError for invalid types, which becomes a 500 error
        with pytest.raises(ValueError, match="Invalid user options"):
            client.post("/v1/crud/configured-models", json=config.model_dump())
