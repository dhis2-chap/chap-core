"""Tests for chapkit integration: model_information propagation and geo serialization."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import chapkit
import httpx
import pytest
from chapkit.api.service_builder import MLServiceInfo
from pydantic import BaseModel
from ulid import ULID

from chap_core.models.chapkit_rest_api_wrapper import CHAPKitRestAPIWrapper, RunInfo
from chap_core.models.external_chapkit_model import (
    ExternalChapkitModel,
    ExternalChapkitModelTemplate,
)
from chap_core.models.utils import _is_chapkit_url

VALID_ULID_1 = str(ULID())
VALID_ULID_2 = str(ULID())

MOCK_INFO_DICT = {
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

MOCK_INFO_RESPONSE = MLServiceInfo.model_validate(MOCK_INFO_DICT)

MOCK_CONFIG_SCHEMA = {
    "$defs": {
        "ModelConfiguration": {
            "properties": {},
        }
    }
}


def _mock_train_predict_response():
    """Create a mock response with valid data for train/predict endpoints."""
    mock = MagicMock()
    mock.json.return_value = {
        "job_id": VALID_ULID_1,
        "artifact_id": VALID_ULID_2,
        "message": "ok",
    }
    return mock


def _make_config_out(config_id=None, name="test"):
    """Create a valid ConfigOut instance."""
    return chapkit.ConfigOut(
        id=config_id or VALID_ULID_1,
        name=name,
        data={"prediction_periods": 3},
        created_at=datetime(2024, 1, 1),
        updated_at=datetime(2024, 1, 1),
    )


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


class TestNameUsesChapkitId:
    def test_name_returns_service_id(self):
        template = ExternalChapkitModelTemplate("http://localhost:8000")

        mock_client = MagicMock()
        mock_client.info.return_value = MOCK_INFO_RESPONSE
        template.client = mock_client

        assert template.name == "test-model"

    def test_name_does_not_use_display_name_format(self):
        template = ExternalChapkitModelTemplate("http://localhost:8000")

        mock_client = MagicMock()
        mock_client.info.return_value = MOCK_INFO_RESPONSE
        template.client = mock_client

        assert template.name != "test_model_v1.0.0"


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

    def test_config_name_uses_service_id(self):
        template = ExternalChapkitModelTemplate("http://localhost:8000")

        mock_client = MagicMock()
        mock_client.info.return_value = MOCK_INFO_RESPONSE
        mock_client.get_config_schema.return_value = MOCK_CONFIG_SCHEMA
        template.client = mock_client
        template.rest_api_url = "http://localhost:8000"

        config = template.get_model_template_config()
        assert config.name == "test-model"

    def test_config_includes_version(self):
        template = ExternalChapkitModelTemplate("http://localhost:8000")

        mock_client = MagicMock()
        mock_client.info.return_value = MOCK_INFO_RESPONSE
        mock_client.get_config_schema.return_value = MOCK_CONFIG_SCHEMA
        template.client = mock_client
        template.rest_api_url = "http://localhost:8000"

        config = template.get_model_template_config()
        assert config.version == "1.0.0"

    def test_config_uses_repository_url_as_source_url(self):
        metadata = dict(MOCK_INFO_DICT["model_metadata"])  # type: ignore[arg-type]
        metadata["repository_url"] = "https://github.com/example/model"
        info_dict = {**MOCK_INFO_DICT, "model_metadata": metadata}
        info = MLServiceInfo.model_validate(info_dict)

        template = ExternalChapkitModelTemplate("http://localhost:8000")
        mock_client = MagicMock()
        mock_client.info.return_value = info
        mock_client.get_config_schema.return_value = MOCK_CONFIG_SCHEMA
        template.client = mock_client
        template.rest_api_url = "http://localhost:8000"

        config = template.get_model_template_config()
        assert config.source_url == "https://github.com/example/model"

    def test_config_falls_back_to_service_url_for_source_url(self):
        template = ExternalChapkitModelTemplate("http://localhost:8000")

        mock_client = MagicMock()
        mock_client.info.return_value = MOCK_INFO_RESPONSE
        mock_client.get_config_schema.return_value = MOCK_CONFIG_SCHEMA
        template.client = mock_client
        template.rest_api_url = "http://localhost:8000"

        config = template.get_model_template_config()
        assert config.source_url == "http://localhost:8000"

    def test_prediction_length_uses_defaults_when_not_specified(self):
        template = ExternalChapkitModelTemplate("http://localhost:8000")

        info_dict = {**MOCK_INFO_DICT}
        del info_dict["min_prediction_periods"]
        del info_dict["max_prediction_periods"]
        info_with_defaults = MLServiceInfo.model_validate(info_dict)

        mock_client = MagicMock()
        mock_client.info.return_value = info_with_defaults
        mock_client.get_config_schema.return_value = MOCK_CONFIG_SCHEMA
        template.client = mock_client
        template.rest_api_url = "http://localhost:8000"
        template._initialized = True

        config = template.get_model_template_config()
        assert config.min_prediction_length == 0
        assert config.max_prediction_length == 100


class TestGetModelPassesModelInformation:
    def test_get_model_passes_model_information(self):
        template = ExternalChapkitModelTemplate("http://localhost:8000")

        config_out = _make_config_out()

        mock_client = MagicMock()
        mock_client.info.return_value = MOCK_INFO_RESPONSE
        mock_client.get_config_schema.return_value = MOCK_CONFIG_SCHEMA
        mock_client.create_config.return_value = config_out
        mock_client.list_configs.return_value = [config_out]
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
        mock_response = _mock_train_predict_response()

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
        mock_response = _mock_train_predict_response()

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
        mock_response = _mock_train_predict_response()

        import pandas as pd

        future = pd.DataFrame({"time_period": ["2024-01"], "location": ["loc1"]})
        run_info = RunInfo(prediction_length=1)

        with patch.object(wrapper, "_request", return_value=mock_response) as mock_req:
            wrapper.predict("artifact-1", future, run_info, geo_features=geo)
            call_kwargs = mock_req.call_args
            body = call_kwargs.kwargs["json"]
            assert isinstance(body["geo"], dict)
            assert body["geo"]["type"] == "FeatureCollection"


class TestTypedResponses:
    @pytest.fixture()
    def wrapper(self):
        return CHAPKitRestAPIWrapper("http://localhost:8000")

    def test_train_returns_typed_response(self, wrapper):
        mock_response = _mock_train_predict_response()

        with patch.object(wrapper, "_request", return_value=mock_response):
            result = wrapper.train(
                "config-1",
                __import__("pandas").DataFrame({"x": [1]}),
                RunInfo(prediction_length=1),
            )
            assert isinstance(result, chapkit.TrainResponse)
            assert result.job_id == VALID_ULID_1
            assert result.artifact_id == VALID_ULID_2

    def test_predict_returns_typed_response(self, wrapper):
        mock_response = _mock_train_predict_response()

        with patch.object(wrapper, "_request", return_value=mock_response):
            result = wrapper.predict(
                "artifact-1",
                __import__("pandas").DataFrame({"x": [1]}),
                RunInfo(prediction_length=1),
            )
            assert isinstance(result, chapkit.PredictResponse)
            assert result.job_id == VALID_ULID_1

    def test_get_job_returns_typed_response(self, wrapper):
        job_ulid = str(ULID())
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": job_ulid,
            "status": "completed",
            "submitted_at": "2024-01-01T00:00:00",
        }

        with patch.object(wrapper, "_request", return_value=mock_response):
            result = wrapper.get_job(job_ulid)
            assert isinstance(result, chapkit.ChapkitJobRecord)
            assert result.status == "completed"

    def test_list_configs_returns_typed_response(self, wrapper):
        ulid1 = str(ULID())
        ulid2 = str(ULID())
        now = "2024-01-01T00:00:00"
        mock_response = MagicMock()
        data = {"prediction_periods": 3}
        mock_response.json.return_value = [
            {"id": ulid1, "name": "config1", "data": data, "created_at": now, "updated_at": now},
            {"id": ulid2, "name": "config2", "data": data, "created_at": now, "updated_at": now},
        ]

        with patch.object(wrapper, "_request", return_value=mock_response):
            result = wrapper.list_configs()
            assert len(result) == 2
            assert all(isinstance(c, chapkit.ConfigOut) for c in result)
            assert result[0].id == ulid1


class TestIsChapkitUrl:
    def test_valid_chapkit_service(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_INFO_DICT

        with patch("chap_core.models.utils.httpx.get", return_value=mock_response):
            assert _is_chapkit_url("http://localhost:8000") is True

    def test_non_chapkit_service(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}

        with patch("chap_core.models.utils.httpx.get", return_value=mock_response):
            assert _is_chapkit_url("http://localhost:8000") is False

    def test_unreachable_service(self):
        with patch("chap_core.models.utils.httpx.get", side_effect=httpx.ConnectError("Connection refused")):
            assert _is_chapkit_url("http://localhost:9999") is False
