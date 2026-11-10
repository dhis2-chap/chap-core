from chap_core.database.model_spec_tables import ModelSpecRead
from chap_core.rest_api.data_models import (
    ConfiguredModelInfoRead,
    ModelConfigurationCreate,
    ModelTemplateRead,
)
from tests.integration.rest_api.db_endpoint_helpers import (
    client,
    get_content,
    logger,
)


def test_list_models_alias_is_gone(clean_engine, dependency_overrides):
    """The /v1/crud/models alias was removed; /configured-models is the only path."""
    response = client.get("/v1/crud/models")
    assert response.status_code == 404, response.text


def test_list_configured_models(celery_session_worker, dependency_overrides):
    response = client.get("/v1/crud/configured-models")
    assert response.status_code == 200, response.json()
    assert isinstance(response.json(), list)
    for m in response.json():
        logger.info(m)
    assert len(response.json()) > 0
    assert "id" in response.json()[0]
    for attr_name in (
        "displayName",
        "id",
        "description",
        "version",
        "sourceDigest",
        "userOptionValues",
        "additionalContinuousCovariates",
    ):
        """Check these here to make sure camelCase in response"""
        assert attr_name in response.json()[0], response.json()[0].keys()
    models = [ModelSpecRead.model_validate(m) for m in response.json()]
    assert "chap_ewars_monthly" in (m.name for m in models)
    ewars_model = next(m for m in models if m.name == "chap_ewars_monthly")
    assert "population" in (f.name for f in ewars_model.covariates)
    assert ewars_model.source_url is not None
    assert ewars_model.source_url.startswith("https:/")
    assert ewars_model.version
    assert isinstance(ewars_model.additional_continuous_covariates, list)


def test_list_model_templates(celery_session_worker, dependency_overrides):
    response = client.get("/v1/crud/model-templates")
    assert response.status_code == 200, response.json()
    assert isinstance(response.json(), list)
    for m in response.json():
        logger.info(m)
    assert len(response.json()) > 0
    assert "id" in response.json()[0]
    for attr_name in ("displayName", "id", "description"):
        """Check these here to make sure camelCase in response"""
        assert attr_name in response.json()[0], response.json()[0].keys()
    models = [ModelTemplateRead.model_validate(m) for m in response.json()]
    assert "chap_ewars_monthly" in (m.name for m in models)
    ewars_model = next(m for m in models if m.name == "chap_ewars_monthly")
    assert "population" in [f for f in ewars_model.required_covariates], ewars_model.required_covariates


def test_add_configured_model_flow(celery_session_worker, dependency_overrides):
    url = "/v1/crud/model-templates"
    content = get_content(url)
    assert isinstance(content, list)
    for m in content:
        logger.info(m)

    model = next(m for m in content if m["name"] == "ewars_template")
    template_id = model["id"]
    print(template_id)
    config = ModelConfigurationCreate(
        name="testing",
        model_template_id=template_id,
        additional_continuous_covariates=["rainfall"],
        user_option_values=dict(precision=2.0, n_lags=3),
    )

    response = client.post("/v1/crud/configured-models", json=config.model_dump())
    assert response.status_code == 200, response.json()


def test_add_configured_model_unknown_template_returns_404(dependency_overrides):
    payload = {"name": "orphan", "modelTemplateId": 999999, "userOptionValues": {}}
    response = client.post("/v1/crud/configured-models", json=payload)
    assert response.status_code == 404, response.text


def test_add_configured_model_without_user_option_values(dependency_overrides):
    """Omitting userOptionValues should not 500; a template with no required user
    options should accept an empty configuration."""
    content = get_content("/v1/crud/model-templates")
    model = next(m for m in content if m["name"] == "naive_model")
    payload = {"name": "naive_without_options", "modelTemplateId": model["id"]}

    response = client.post("/v1/crud/configured-models", json=payload)
    assert response.status_code == 200, response.json()
    assert response.json()["userOptionValues"] == {}


def test_get_configured_model_info(celery_session_worker, dependency_overrides):
    configured = get_content("/v1/crud/configured-models")
    default = next(m for m in configured if m["name"] == "chap_ewars_monthly")

    response = client.get(f"/v1/crud/configured-models/{default['id']}")
    assert response.status_code == 200, response.json()
    body = response.json()
    for key in ("id", "name", "displayName", "modelTemplateId", "modelTemplate"):
        assert key in body, body.keys()
    info = ConfiguredModelInfoRead.model_validate(body)
    assert info.name == "chap_ewars_monthly"

    missing = client.get("/v1/crud/configured-models/999999")
    assert missing.status_code == 404, missing.json()
