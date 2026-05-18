"""
Tests for chap__covid_mask handling through the configure-model endpoint, using
the EWARS template as a representative model template with required covariates
and additional user options.

Related: CLIM-191 (ChapUserOptions / covid_mask).
"""

import logging

from fastapi.testclient import TestClient

from chap_core.rest_api.app import app
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


class TestEwarsCovidMaskConfiguration:
    """Tests for chap__covid_mask option flowing through the EWARS template."""

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
        Regression scenario: bug report mentioned a barebone variant with
        n_lags=3, precision=0.01, covariates rainfall + mean_temperature.
        chap__covid_mask=false should round-trip cleanly alongside those.
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

        assert configured_model["userOptionValues"]["n_lags"] == 3
        assert configured_model["userOptionValues"]["precision"] == 0.01
        assert configured_model["userOptionValues"]["chap__covid_mask"] is False
        assert set(configured_model["additionalContinuousCovariates"]) == {
            "rainfall",
            "mean_temperature",
        }
