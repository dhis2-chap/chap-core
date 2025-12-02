import logging
import pytest
from fastapi.testclient import TestClient

from chap_core.rest_api.v1.rest_api import app
from chap_core.rest_api.v1.routers.crud import ModelConfigurationCreate

logger = logging.getLogger(__name__)
client = TestClient(app)


def get_content(url):
    response = client.get(url)
    assert response.status_code == 200, response.json()
    return response.json()


def test_list_model_templates_includes_chap_options(celery_session_worker, dependency_overrides):
    """Test that /crud/model-templates endpoint includes chap__ options."""
    response = client.get("/v1/crud/model-templates")
    assert response.status_code == 200, response.json()
    templates = response.json()
    assert len(templates) > 0

    # Check that at least one template has chap__ options
    template = templates[0]
    assert "userOptions" in template
    user_options = template["userOptions"]

    # Should have chap__covid_mask option
    assert "chap__covid_mask" in user_options
    assert user_options["chap__covid_mask"]["type"] == "boolean"
    assert user_options["chap__covid_mask"]["default"] is False


def test_create_configured_model_without_chap_options(celery_session_worker, dependency_overrides):
    """Test creating a configured model without chap__ options (backward compatibility)."""
    url = "/v1/crud/model-templates"
    content = get_content(url)

    model = next(m for m in content if m["name"] == "ewars_template")
    template_id = model["id"]

    config = ModelConfigurationCreate(
        name="test_without_chap_options",
        model_template_id=template_id,
        additional_continuous_covariates=["rainfall"],
        user_option_values={"precision": 2.0, "n_lags": 3},
    )

    response = client.post("/v1/crud/configured-models", json=config.model_dump())
    assert response.status_code == 200, response.json()
    configured_model = response.json()
    assert "test_without_chap_options" in configured_model["name"]


def test_create_configured_model_with_chap_covid_mask_false(celery_session_worker, dependency_overrides):
    """Test creating a configured model with chap__covid_mask set to False."""
    url = "/v1/crud/model-templates"
    content = get_content(url)

    model = next(m for m in content if m["name"] == "ewars_template")
    template_id = model["id"]

    config = ModelConfigurationCreate(
        name="test_with_chap_covid_mask_false",
        model_template_id=template_id,
        additional_continuous_covariates=["rainfall"],
        user_option_values={"precision": 2.0, "n_lags": 3, "chap__covid_mask": False},
    )

    response = client.post("/v1/crud/configured-models", json=config.model_dump())
    assert response.status_code == 200, response.json()
    configured_model = response.json()
    assert "test_with_chap_covid_mask_false" in configured_model["name"]
    assert configured_model["userOptionValues"]["chap__covid_mask"] is False


def test_create_configured_model_with_chap_covid_mask_true(celery_session_worker, dependency_overrides):
    """Test creating a configured model with chap__covid_mask set to True."""
    url = "/v1/crud/model-templates"
    content = get_content(url)

    model = next(m for m in content if m["name"] == "ewars_template")
    template_id = model["id"]

    config = ModelConfigurationCreate(
        name="test_with_chap_covid_mask_true",
        model_template_id=template_id,
        additional_continuous_covariates=["rainfall"],
        user_option_values={"precision": 2.0, "n_lags": 3, "chap__covid_mask": True},
    )

    response = client.post("/v1/crud/configured-models", json=config.model_dump())
    assert response.status_code == 200, response.json()
    configured_model = response.json()
    assert "test_with_chap_covid_mask_true" in configured_model["name"]
    assert configured_model["userOptionValues"]["chap__covid_mask"] is True


def test_create_configured_model_mixed_options(celery_session_worker, dependency_overrides):
    """Test creating a configured model with both model-specific and chap options."""
    url = "/v1/crud/model-templates"
    content = get_content(url)

    model = next(m for m in content if m["name"] == "ewars_template")
    template_id = model["id"]

    config = ModelConfigurationCreate(
        name="test_mixed_options",
        model_template_id=template_id,
        additional_continuous_covariates=["rainfall", "mean_temperature"],
        user_option_values={"precision": 1.5, "n_lags": 5, "chap__covid_mask": True},
    )

    response = client.post("/v1/crud/configured-models", json=config.model_dump())
    assert response.status_code == 200, response.json()
    configured_model = response.json()
    assert "test_mixed_options" in configured_model["name"]
    assert configured_model["userOptionValues"]["precision"] == 1.5
    assert configured_model["userOptionValues"]["n_lags"] == 5
    assert configured_model["userOptionValues"]["chap__covid_mask"] is True
