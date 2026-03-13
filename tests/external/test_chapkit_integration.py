"""Tests for chapkit integration: model_information propagation and geo serialization."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from chap_core.models.chapkit_rest_api_wrapper import CHAPKitRestAPIWrapper, RunInfo
from chap_core.models.external_chapkit_model import (
    ExternalChapkitModel,
    ExternalChapkitModelTemplate,
)


MOCK_INFO_RESPONSE = {
    "id": "test-model",
    "display_name": "Test Model",
    "version": "1.0.0",
    "description": "A test model",
    "model_metadata": {
        "author": "Test",
        "author_assessed_status": "yellow",
    },
    "period_type": "monthly",
    "min_prediction_periods": 1,
    "max_prediction_periods": 12,
    "allow_free_additional_continuous_covariates": False,
    "required_covariates": [],
    "requires_geo": False,
}

MOCK_CONFIG_SCHEMA = {
    "$defs": {
        "ModelConfiguration": {
            "properties": {},
        }
    }
}


class TestExternalChapkitModelInformation:
    def test_model_has_model_information_when_provided(self):
        mock_config = MagicMock()
        mock_config.min_prediction_length = 1
        mock_config.max_prediction_length = 12
        model = ExternalChapkitModel("test", "http://localhost:8000", "config-id", model_information=mock_config)
        assert model.model_information is mock_config
        assert model.model_information.min_prediction_length == 1
        assert model.model_information.max_prediction_length == 12

    def test_model_information_defaults_to_none(self):
        model = ExternalChapkitModel("test", "http://localhost:8000", "config-id")
        assert model.model_information is None


class TestGetModelTemplateConfig:
    def test_maps_prediction_periods_to_prediction_length(self):
        template = ExternalChapkitModelTemplate("http://localhost:8000")

        mock_client = MagicMock()
        mock_client.info.return_value = MOCK_INFO_RESPONSE
        mock_client.get_config_schema.return_value = MOCK_CONFIG_SCHEMA
        template.client = mock_client
        template.rest_api_url = "http://localhost:8000"
        template._initialized = True

        config = template.get_model_template_config()
        assert config.min_prediction_length == 1
        assert config.max_prediction_length == 12

    def test_prediction_length_none_when_not_in_info(self):
        template = ExternalChapkitModelTemplate("http://localhost:8000")

        info_without_periods = {**MOCK_INFO_RESPONSE}
        del info_without_periods["min_prediction_periods"]
        del info_without_periods["max_prediction_periods"]

        mock_client = MagicMock()
        mock_client.info.return_value = info_without_periods
        mock_client.get_config_schema.return_value = MOCK_CONFIG_SCHEMA
        template.client = mock_client
        template.rest_api_url = "http://localhost:8000"
        template._initialized = True

        config = template.get_model_template_config()
        assert config.min_prediction_length is None
        assert config.max_prediction_length is None


class TestGetModelPassesModelInformation:
    def test_get_model_passes_model_information(self):
        template = ExternalChapkitModelTemplate("http://localhost:8000")

        mock_client = MagicMock()
        mock_client.info.return_value = MOCK_INFO_RESPONSE
        mock_client.get_config_schema.return_value = MOCK_CONFIG_SCHEMA
        mock_client.create_config.return_value = {"id": "config-123"}
        mock_client.list_configs.return_value = [{"id": "config-123"}]
        template.client = mock_client
        template.rest_api_url = "http://localhost:8000"
        template._initialized = True

        model = template.get_model({})
        assert model.model_information is not None
        assert model.model_information.min_prediction_length == 1
        assert model.model_information.max_prediction_length == 12


class FakeGeoModel(BaseModel):
    type: str = "FeatureCollection"
    features: list = []


class TestGeoSerialization:
    @pytest.fixture()
    def wrapper(self):
        return CHAPKitRestAPIWrapper("http://localhost:8000")

    def test_train_serializes_pydantic_geo(self, wrapper):
        geo = FakeGeoModel()
        mock_response = MagicMock()
        mock_response.json.return_value = {"job_id": "j1", "artifact_id": "a1"}

        import pandas as pd

        data = pd.DataFrame({"time_period": ["2024-01"], "location": ["loc1"], "disease_cases": [10]})
        run_info = RunInfo(prediction_length=1)

        with patch.object(wrapper, "_request", return_value=mock_response) as mock_req:
            wrapper.train("config-1", data, run_info, geo_features=geo)
            call_kwargs = mock_req.call_args
            body = call_kwargs.kwargs["json"]
            assert isinstance(body["geo"], dict)
            assert body["geo"]["type"] == "FeatureCollection"

    def test_train_passes_dict_geo_unchanged(self, wrapper):
        geo = {"type": "FeatureCollection", "features": []}
        mock_response = MagicMock()
        mock_response.json.return_value = {"job_id": "j1", "artifact_id": "a1"}

        import pandas as pd

        data = pd.DataFrame({"time_period": ["2024-01"], "location": ["loc1"], "disease_cases": [10]})
        run_info = RunInfo(prediction_length=1)

        with patch.object(wrapper, "_request", return_value=mock_response) as mock_req:
            wrapper.train("config-1", data, run_info, geo_features=geo)
            call_kwargs = mock_req.call_args
            body = call_kwargs.kwargs["json"]
            assert body["geo"] is geo

    def test_predict_serializes_pydantic_geo(self, wrapper):
        geo = FakeGeoModel()
        mock_response = MagicMock()
        mock_response.json.return_value = {"job_id": "j1", "artifact_id": "a1"}

        import pandas as pd

        future = pd.DataFrame({"time_period": ["2024-01"], "location": ["loc1"]})
        run_info = RunInfo(prediction_length=1)

        with patch.object(wrapper, "_request", return_value=mock_response) as mock_req:
            wrapper.predict("artifact-1", future, run_info, geo_features=geo)
            call_kwargs = mock_req.call_args
            body = call_kwargs.kwargs["json"]
            assert isinstance(body["geo"], dict)
            assert body["geo"]["type"] == "FeatureCollection"
