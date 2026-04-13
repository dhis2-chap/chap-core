from unittest.mock import patch

import httpx
import pytest

from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.exceptions import ChapkitServiceStartupError
from chap_core.file_io.example_data_set import datasets
from chap_core.models.chapkit_service_manager import (
    ChapkitServiceManager,
    find_available_port,
    is_url,
)
from chap_core.models.external_chapkit_model import ExternalChapkitModelTemplate

model_url = "http://localhost:8003"


@pytest.fixture
def dataset():
    dataset_name = "ISIMIP_dengue_harmonized"
    example_dataset = datasets[dataset_name]
    loaded_dataset = example_dataset.load()
    return loaded_dataset["brazil"]


@pytest.fixture
def service_available():
    try:
        response = httpx.get(model_url + "/health", timeout=2)
        if response.status_code != 200:
            pytest.skip(f"Service not available at {model_url}")
    except:
        pytest.skip(f"Service not available at {model_url}")

    return model_url


@pytest.mark.skip(reason="Needs a running chapkit model service")
def test_external_chapkit_model_basic(service_available, dataset):
    train, test = train_test_generator(dataset, 3, 2)
    historic, future, truth = next(test)  # type: ignore[reportArgumentType]

    template = ExternalChapkitModelTemplate(service_available)
    # model = template.get_model({"user_option_values": {"max_epochs": 2}})
    model = template.get_model({})
    id = model.train(historic)
    prediction = model.predict(historic, future)
    print("PREDICTION")
    print(prediction)
    prediction_df = prediction.to_pandas()

    print(prediction_df)
    assert len(prediction_df) == len(future.to_pandas())


@pytest.mark.skip(reason="Needs a running chapkit model service")
def test_chapkit_model_wrapping(service_available):
    # just test that we can wrap the info in the old object
    template = ExternalChapkitModelTemplate(service_available)
    config = template.get_model_template_config()
    print(config)


class TestIsUrl:
    def test_http_url(self):
        assert is_url("http://localhost:8000") is True

    def test_https_url(self):
        assert is_url("https://example.com/model") is True

    def test_directory_path(self):
        assert is_url("/path/to/model") is False

    def test_relative_path(self):
        assert is_url("./models/my_model") is False


class TestFindAvailablePort:
    def test_finds_port(self):
        port = find_available_port(start_port=10000)
        assert 10000 <= port < 10100

    def test_raises_when_no_ports_available(self):
        with patch("socket.socket") as mock_socket:
            mock_instance = mock_socket.return_value.__enter__.return_value
            mock_instance.bind.side_effect = OSError
            with pytest.raises(ChapkitServiceStartupError):
                find_available_port(start_port=10000, max_attempts=5)


class TestChapkitServiceManager:
    def test_validates_nonexistent_directory(self, tmp_path):
        manager = ChapkitServiceManager(str(tmp_path / "nonexistent"))
        with pytest.raises(ChapkitServiceStartupError, match="does not exist"):
            manager._validate_directory()

    def test_url_property_before_start(self, tmp_path):
        manager = ChapkitServiceManager(str(tmp_path))
        with pytest.raises(RuntimeError, match="Service not started"):
            _ = manager.url


class TestGetModelTemplateConfig:
    """Test that get_model_template_config extracts user_options from the chapkit config schema."""

    def _make_template_with_schema(self, schema_response):
        """Create a template with mocked client returning the given schema."""
        from unittest.mock import MagicMock

        from chap_core.rest_api.services.schemas import MLServiceInfo

        template = ExternalChapkitModelTemplate("http://fake:8000")
        template._is_url_mode = True
        template.rest_api_url = "http://fake:8000"

        mock_client = MagicMock()
        mock_client.info.return_value = MLServiceInfo(
            id="test-model",
            display_name="Test Model",
            version="1.0.0",
            description="Test",
            model_metadata={},
            period_type="monthly",
            required_covariates=["population"],
        )
        mock_client.get_config_schema.return_value = schema_response
        template.client = mock_client
        return template

    def test_extracts_user_options_from_top_level_properties(self):
        """Chapkit services expose config schema at the top level."""
        schema = {
            "properties": {
                "prediction_periods": {"type": "integer", "default": 3},
                "additional_continuous_covariates": {"type": "array", "items": {"type": "string"}},
                "n_lags": {"type": "integer", "default": 3, "description": "Number of lags"},
                "precision": {"type": "number", "default": 0.01, "description": "Precision"},
            },
            "title": "EwarsConfig",
            "type": "object",
        }
        template = self._make_template_with_schema(schema)
        config = template.get_model_template_config()
        assert config.user_options is not None
        assert "n_lags" in config.user_options
        assert "precision" in config.user_options
        assert config.user_options["n_lags"]["default"] == 3
        assert config.user_options["precision"]["default"] == 0.01

    def test_filters_reserved_fields(self):
        """prediction_periods and additional_continuous_covariates are reserved."""
        schema = {
            "properties": {
                "prediction_periods": {"type": "integer", "default": 3},
                "additional_continuous_covariates": {"type": "array"},
                "n_lags": {"type": "integer", "default": 3},
            },
            "type": "object",
        }
        template = self._make_template_with_schema(schema)
        config = template.get_model_template_config()
        assert config.user_options is not None
        assert "prediction_periods" not in config.user_options
        assert "additional_continuous_covariates" not in config.user_options
        assert "n_lags" in config.user_options

    def test_falls_back_to_defs_path(self):
        """Legacy MLflow-era models store schema under $defs.ModelConfiguration."""
        schema = {
            "$defs": {
                "ModelConfiguration": {
                    "properties": {
                        "max_epochs": {"type": "integer", "default": 10},
                    }
                }
            }
        }
        template = self._make_template_with_schema(schema)
        config = template.get_model_template_config()
        assert config.user_options is not None
        assert "max_epochs" in config.user_options

    def test_empty_schema_returns_empty_user_options(self):
        """No properties at all."""
        template = self._make_template_with_schema({})
        config = template.get_model_template_config()
        assert config.user_options == {}


class TestExternalChapkitModelTemplateDetection:
    def test_detects_url_mode(self):
        template = ExternalChapkitModelTemplate("http://localhost:8000")
        assert template.is_url_mode is True

    def test_detects_directory_mode(self, tmp_path):
        template = ExternalChapkitModelTemplate(str(tmp_path))
        assert template.is_url_mode is False


class TestExternalChapkitModelTemplateDirectoryMode:
    def test_raises_without_context_manager_is_healthy(self, tmp_path):
        template = ExternalChapkitModelTemplate(str(tmp_path))
        with pytest.raises(RuntimeError, match="context manager"):
            template.is_healthy()

    def test_raises_without_context_manager_get_model(self, tmp_path):
        template = ExternalChapkitModelTemplate(str(tmp_path))
        with pytest.raises(RuntimeError, match="context manager"):
            template.get_model({})

    def test_raises_without_context_manager_name(self, tmp_path):
        template = ExternalChapkitModelTemplate(str(tmp_path))
        with pytest.raises(RuntimeError, match="context manager"):
            _ = template.name
