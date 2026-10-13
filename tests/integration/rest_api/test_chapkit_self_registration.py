"""End-to-end tests for chapkit self-registration and template registration.

A registered service gets a model template, stored from its own info and schema,
and a health status on GET /v1/crud/model-templates. It gets no configured models:
those come from the marketplace entry through POST /v1/crud/configured-models,
which is what chap-admin install calls after storing the template of the service it
started through POST /v1/crud/model-templates/from-service.
"""

import logging
from unittest.mock import MagicMock, patch

import fakeredis
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, select

import chap_core.database.tables  # noqa: F401 - ensure all table models are registered with SQLModel
from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB, ModelTemplateDB, ModelTemplateRole
from chap_core.rest_api.app import app
from chap_core.rest_api.services.orchestrator import Orchestrator
from chap_core.rest_api.services.schemas import MLServiceInfo, RegistrationRequest
from chap_core.rest_api.v1.routers.dependencies import get_session
from chap_core.rest_api.v2.dependencies import get_orchestrator

MOCK_INFO_DICT = {
    "id": "test-model",
    "display_name": "Test Model",
    "version": "1.0.0",
    "git_revision": "a" * 40,
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

SCHEMA = {
    "properties": {
        "n_lags": {"type": "integer", "default": 3},
        "prediction_periods": {"type": "integer", "default": 1},
    }
}


@pytest.fixture
def fake_orchestrator():
    return Orchestrator(redis_client=fakeredis.FakeRedis())


@pytest.fixture
def db_engine():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def mock_wrapper_cls():
    cls = MagicMock()
    cls.return_value.get_config_schema.return_value = SCHEMA
    return cls


@pytest.fixture
def client(db_engine, fake_orchestrator, mock_wrapper_cls):
    def get_test_session():
        with Session(db_engine) as session:
            yield session

    app.dependency_overrides[get_session] = get_test_session
    # The v2 register endpoint takes the orchestrator through Depends, while the lazy
    # v1 template sync calls the factory directly, so both need the fake.
    app.dependency_overrides[get_orchestrator] = lambda: fake_orchestrator

    with (
        patch("chap_core.rest_api.v2.dependencies.get_orchestrator", return_value=fake_orchestrator),
        patch("chap_core.models.chapkit_rest_api_wrapper.CHAPKitRestAPIWrapper", mock_wrapper_cls),
    ):
        yield TestClient(app, raise_server_exceptions=False)

    app.dependency_overrides.clear()


@pytest.fixture
def register_service(fake_orchestrator):
    def _register(info_dict=None):
        info_dict = info_dict or MOCK_INFO_DICT
        info = MLServiceInfo.model_validate(info_dict)
        request = RegistrationRequest(url="http://test-service:8080", info=info)
        fake_orchestrator.register(request)

    return _register


def _install(client, register_service, **info):
    """What chap-admin install does once the service it started is up: store its template."""
    register_service({**MOCK_INFO_DICT, **info})
    response = client.post("/v1/crud/model-templates/from-service", json={"serviceId": "test-model"})
    assert response.status_code == 200, response.text
    return response.json()


def _test_model(client):
    matching = [t for t in client.get("/v1/crud/model-templates").json() if t["name"] == "test-model"]
    assert len(matching) == 1
    return matching[0]


def test_install_sets_the_role_from_the_marketplace_entry(client, register_service):
    # Registration already stores the template, without a role.
    register_service()
    assert _test_model(client)["role"] is None

    response = client.post(
        "/v1/crud/model-templates/from-service", json={"serviceId": "test-model", "role": "comparison"}
    )
    assert response.status_code == 200, response.text
    assert response.json()["role"] == "comparison"
    assert _test_model(client)["role"] == "comparison"

    # An entry that drops the flag makes it an ordinary model again.
    assert _install(client, register_service)["role"] is None


def test_install_cannot_mark_a_model_as_baseline(client, register_service):
    register_service()
    response = client.post(
        "/v1/crud/model-templates/from-service", json={"serviceId": "test-model", "role": "baseline"}
    )
    assert response.status_code == 422


@pytest.mark.parametrize("version", ["1.0.0", "2.0.0"])
def test_a_service_cannot_store_or_supersede_a_baseline(client, register_service, db_engine, version):
    with Session(db_engine) as session:
        session.add(
            ModelTemplateDB(name="test-model", version="1.0.0", source_digest="a" * 40, role=ModelTemplateRole.baseline)
        )
        session.commit()
    register_service({**MOCK_INFO_DICT, "version": version})

    response = client.post("/v1/crud/model-templates/from-service", json={"serviceId": "test-model"})

    assert response.status_code == 409
    with Session(db_engine) as session:
        templates = session.exec(select(ModelTemplateDB).where(ModelTemplateDB.name == "test-model")).all()
        assert [(t.version, t.role) for t in templates] == [("1.0.0", ModelTemplateRole.baseline)]


def test_registered_service_becomes_a_template_without_configured_models(client, register_service):
    register_service({**MOCK_INFO_DICT, "git_revision": "b" * 40})

    template = _test_model(client)
    assert template["healthStatus"] == "live"
    assert template["sourceDigest"] == "b" * 40
    assert template["usesChapkit"] is True
    assert template["userOptions"] == {"n_lags": {"type": "integer", "default": 3}}
    # Configurations come from the marketplace entry, never from the service.
    assert client.get("/v1/crud/configured-models").json() == []


def test_registration_endpoint_stores_the_template_eagerly(client):
    response = client.post("/v2/services/$register", json={"url": "http://test-service:8080", "info": MOCK_INFO_DICT})
    assert response.status_code == 200
    assert _test_model(client)["version"] == "1.0.0"


@pytest.mark.parametrize("git_revision", [None, ""])
def test_service_without_a_git_revision_is_not_stored_until_it_reports_one(client, register_service, git_revision):
    """Storing a row without a digest would burn the version label, so the label stays free."""
    register_service({**MOCK_INFO_DICT, "git_revision": git_revision})
    assert client.get("/v1/crud/model-templates").json() == []

    register_service({**MOCK_INFO_DICT, "git_revision": "a" * 40})
    template = _test_model(client)
    assert template["sourceDigest"] == "a" * 40
    assert template["healthStatus"] == "live"


def test_installed_template_is_reused_when_its_service_registers_again(client, register_service, mock_wrapper_cls):
    installed = _install(client, register_service)
    assert installed["sourceDigest"] == "a" * 40
    assert installed["usesChapkit"] is True
    assert installed["userOptions"] == {"n_lags": {"type": "integer", "default": 3}}

    register_service()
    template = _test_model(client)
    assert template["healthStatus"] == "live"
    assert template["id"] == installed["id"]
    # A stored version is write-once, so nothing is read from the service for it again.
    assert mock_wrapper_cls.call_count == 1


def test_sync_is_idempotent(client, register_service, mock_wrapper_cls):
    register_service()
    first = client.get("/v1/crud/model-templates").json()
    assert client.get("/v1/crud/model-templates").json() == first
    assert mock_wrapper_cls.call_count == 1


def test_re_registered_service_with_new_version_adds_a_new_template(
    client, register_service, fake_orchestrator, db_engine
):
    register_service()
    first_id = _test_model(client)["id"]

    fake_orchestrator.deregister("test-model")
    register_service({**MOCK_INFO_DICT, "version": "2.0.0", "display_name": "Updated Model", "requires_geo": True})

    # The stored version stays live until the new one can run, that is, has a configuration.
    assert _test_model(client)["version"] == "1.0.0"
    with Session(db_engine) as session:
        new_id = session.exec(select(ModelTemplateDB.id).where(ModelTemplateDB.version == "2.0.0")).one()
    client.post("/v1/crud/configured-models", json={"name": "default", "modelTemplateId": new_id})
    live = _test_model(client)
    assert (live["version"], live["displayName"], live["requiresGeo"], live["isLive"]) == (
        "2.0.0",
        "Updated Model",
        True,
        True,
    )
    assert live["id"] != first_id
    with Session(db_engine) as session:
        superseded = session.get(ModelTemplateDB, first_id)
        assert superseded is not None
        assert superseded.is_live is False


def test_installing_another_revision_under_a_stored_version_is_refused(client, register_service):
    _install(client, register_service)
    register_service({**MOCK_INFO_DICT, "git_revision": "b" * 40})
    response = client.post("/v1/crud/model-templates/from-service", json={"serviceId": "test-model"})
    assert response.status_code == 409
    assert "b" * 40 in response.json()["detail"]
    assert _test_model(client)["sourceDigest"] == "a" * 40


def test_a_new_version_is_a_new_live_row_and_the_old_one_stays(client, register_service, db_engine):
    first_id = _install(client, register_service)["id"]
    template = _install(client, register_service, version="1.0.1", git_revision="b" * 40)
    assert template["id"] != first_id
    # The stored version stays live until the new one can run, that is, has a configuration.
    assert _test_model(client)["version"] == "1.0.0"
    client.post("/v1/crud/configured-models", json={"name": "default", "modelTemplateId": template["id"]})
    assert _test_model(client)["version"] == "1.0.1"
    with Session(db_engine) as session:
        superseded = session.get(ModelTemplateDB, first_id)
        assert superseded is not None
        assert superseded.is_live is False


@pytest.mark.parametrize("reported", ["b" * 40, None])
def test_republished_service_under_the_same_version_is_a_revision_mismatch(
    client, register_service, mock_wrapper_cls, reported
):
    _install(client, register_service)
    register_service({**MOCK_INFO_DICT, "git_revision": reported})
    template = _test_model(client)
    assert template["healthStatus"] == "revision_mismatch"
    assert template["sourceDigest"] == "a" * 40
    assert template["archived"] is False
    # Nothing is fetched from a mismatched service.
    assert mock_wrapper_cls.call_count == 1


def test_registration_response_tells_a_service_without_a_git_revision_what_to_do(client):
    payload = {"url": "http://test-service:8080", "info": {**MOCK_INFO_DICT, "git_revision": None}}
    response = client.post("/v2/services/$register", json=payload)

    assert response.status_code == 200
    assert "GIT_REVISION build arg" in response.json()["message"]


def test_revision_mismatch_of_another_version_does_not_flag_the_live_template(client, register_service):
    register_service()
    assert _test_model(client)["healthStatus"] == "live"

    # Another version of the same model, which conflicts, takes over the registration.
    register_service({**MOCK_INFO_DICT, "git_revision": None, "version": "1.0.1"})
    template = _test_model(client)
    assert template["version"] == "1.0.0"
    assert template["healthStatus"] is None


def test_registration_response_reports_a_revision_mismatch(client, register_service):
    _install(client, register_service)
    payload = {"url": "http://test-service:8080", "info": {**MOCK_INFO_DICT, "git_revision": "b" * 40}}
    response = client.post("/v2/services/$register", json=payload)

    assert response.status_code == 200
    message = response.json()["message"]
    assert "version '1.0.0'" in message
    assert "a" * 40 in message and "b" * 40 in message
    assert "info.version" in message


def test_redeploying_the_stored_revision_clears_the_mismatch(client, register_service):
    _install(client, register_service)
    register_service({**MOCK_INFO_DICT, "git_revision": "b" * 40})
    assert _test_model(client)["healthStatus"] == "revision_mismatch"

    # The mismatch is computed at read time, so the right image needs no cleanup.
    register_service()
    assert _test_model(client)["healthStatus"] == "live"


def test_schema_fetch_failure_does_not_freeze_empty_user_options(client, register_service, mock_wrapper_cls, caplog):
    mock_wrapper_cls.return_value.get_config_schema.side_effect = [
        ConnectionError("service temporarily unavailable"),
        SCHEMA,
    ]
    register_service()

    # An incomplete template is not created while the schema endpoint is down.
    with caplog.at_level(logging.WARNING, logger="chap_core.rest_api.v1.routers.crud"):
        assert client.get("/v1/crud/model-templates").json() == []
    assert any(
        record.levelno == logging.WARNING
        and "Could not fetch config schema" in record.message
        and "will retry next sync" in record.message
        and record.exc_info is not None
        for record in caplog.records
    )
    assert _test_model(client)["userOptions"] == {"n_lags": {"type": "integer", "default": 3}}


def test_deregistered_service_loses_live_status_but_the_template_stays(client, register_service, fake_orchestrator):
    _install(client, register_service)
    assert _test_model(client)["healthStatus"] == "live"

    fake_orchestrator.deregister("test-model")

    template = _test_model(client)
    assert template["healthStatus"] is None
    assert template["archived"] is False


def test_retiring_a_template_hides_it_and_its_configured_models_until_it_is_installed_again(client, register_service):
    template_id = _install(client, register_service)["id"]
    response = client.post(
        "/v1/crud/configured-models",
        json={"name": "tuned", "modelTemplateId": template_id, "userOptionValues": {"n_lags": 5}},
    )
    assert response.status_code == 200
    assert [m["name"] for m in client.get("/v1/crud/configured-models").json()] == ["test-model:tuned"]

    assert client.delete(f"/v1/crud/model-templates/{template_id}").status_code == 200
    assert _test_model(client)["archived"] is True
    assert client.get("/v1/crud/configured-models").json() == []
    assert client.delete("/v1/crud/model-templates/999").status_code == 404

    # Installing again shows the template, and re-adding a configuration shows it too.
    assert _install(client, register_service)["archived"] is False
    client.post(
        "/v1/crud/configured-models",
        json={"name": "tuned", "modelTemplateId": template_id, "userOptionValues": {"n_lags": 5}},
    )
    assert [m["name"] for m in client.get("/v1/crud/configured-models").json()] == ["test-model:tuned"]


@pytest.mark.parametrize(
    "service_state, expected_health",
    [
        ("live", "live"),
        ("revision_mismatch", "revision_mismatch"),
        ("other_version", None),
        ("deregistered", None),
        ("registry_unavailable", None),
        ("non_chapkit", None),
    ],
)
def test_configured_model_health_comes_from_its_template_without_syncing(
    client, register_service, fake_orchestrator, mock_wrapper_cls, db_engine, service_state, expected_health
):
    template = _install(client, register_service)
    client.post("/v1/crud/configured-models", json={"name": "custom-config", "modelTemplateId": template["id"]})
    service_calls = mock_wrapper_cls.call_count

    if service_state == "revision_mismatch":
        register_service({**MOCK_INFO_DICT, "git_revision": "b" * 40})
    elif service_state == "other_version":
        register_service({**MOCK_INFO_DICT, "version": "1.0.1", "git_revision": None})
    elif service_state == "deregistered":
        fake_orchestrator.deregister("test-model")
    elif service_state == "registry_unavailable":
        fake_orchestrator.get_all = MagicMock(side_effect=ConnectionError("registry unavailable"))
    elif service_state == "non_chapkit":
        with Session(db_engine) as session:
            stored_template = session.get(ModelTemplateDB, template["id"])
            assert stored_template is not None
            stored_template.uses_chapkit = False
            session.commit()

    response = client.get("/v1/crud/configured-models")

    assert response.status_code == 200
    models = response.json()
    assert len(models) == 1
    assert models[0]["healthStatus"] == expected_health
    assert mock_wrapper_cls.call_count == service_calls
    with Session(db_engine) as session:
        stored_template = session.get(ModelTemplateDB, template["id"])
        assert stored_template is not None
        assert stored_template.archived is False


def test_retiring_the_live_version_hands_live_status_back_to_the_previous_one(client, register_service):
    first_id = _install(client, register_service)["id"]
    client.post("/v1/crud/configured-models", json={"name": "default", "modelTemplateId": first_id})
    second_id = _install(client, register_service, version="1.0.1", git_revision="b" * 40)["id"]
    client.post("/v1/crud/configured-models", json={"name": "default", "modelTemplateId": second_id})
    assert _test_model(client)["version"] == "1.0.1"

    assert client.delete(f"/v1/crud/model-templates/{second_id}").status_code == 200
    live = _test_model(client)
    assert (live["version"], live["archived"]) == ("1.0.0", False)
    assert [m["name"] for m in client.get("/v1/crud/configured-models").json()] == ["test-model"]


def test_retiring_all_versions_leaves_no_version_live(client, register_service):
    first_id = _install(client, register_service)["id"]
    client.post("/v1/crud/configured-models", json={"name": "default", "modelTemplateId": first_id})
    second_id = _install(client, register_service, version="1.0.1", git_revision="b" * 40)["id"]
    client.post("/v1/crud/configured-models", json={"name": "default", "modelTemplateId": second_id})

    assert client.delete(f"/v1/crud/model-templates/{second_id}?allVersions=true").status_code == 200
    template = _test_model(client)
    assert (template["version"], template["archived"]) == ("1.0.1", True)
    assert client.get("/v1/crud/configured-models").json() == []


def test_template_archived_by_an_earlier_chap_comes_back_when_its_service_registers(
    client, register_service, db_engine
):
    """Earlier versions archived a template whose service went away and left its configured models."""
    with Session(db_engine) as session:
        template = ModelTemplateDB(
            name="test-model", version="1.0.0", source_digest="a" * 40, uses_chapkit=True, archived=True, is_live=True
        )
        session.add(template)
        session.commit()
        session.add(ConfiguredModelDB(name="test-model", model_template_id=template.id, uses_chapkit=True))
        session.commit()

    register_service()
    assert _test_model(client)["archived"] is False
    assert [m["name"] for m in client.get("/v1/crud/configured-models").json()] == ["test-model"]


def test_retired_template_stays_retired_while_its_service_runs(client, register_service):
    template_id = _install(client, register_service)["id"]
    client.post("/v1/crud/configured-models", json={"name": "default", "modelTemplateId": template_id})
    assert client.delete(f"/v1/crud/model-templates/{template_id}").status_code == 200

    register_service()
    assert _test_model(client)["archived"] is True
    assert client.get("/v1/crud/configured-models").json() == []


def test_template_from_a_registered_service(client, register_service):
    register_service()
    response = client.post("/v1/crud/model-templates/from-service", json={"serviceId": "test-model"})
    assert response.status_code == 200, response.text
    template = response.json()
    assert (template["name"], template["version"], template["sourceDigest"]) == ("test-model", "1.0.0", "a" * 40)
    assert template["usesChapkit"] is True
    assert template["userOptions"] == {"n_lags": {"type": "integer", "default": 3}}
    assert _test_model(client)["healthStatus"] == "live"
    # Repeating the call returns the stored row.
    assert (
        client.post("/v1/crud/model-templates/from-service", json={"serviceId": "test-model"}).json()["id"]
        == (template["id"])
    )


def test_template_from_an_unknown_or_unversioned_service_is_refused(client, register_service):
    assert client.post("/v1/crud/model-templates/from-service", json={"serviceId": "test-model"}).status_code == 404
    register_service({**MOCK_INFO_DICT, "git_revision": None})
    response = client.post("/v1/crud/model-templates/from-service", json={"serviceId": "test-model"})
    assert response.status_code == 409
    assert "GIT_REVISION build arg" in response.json()["detail"]
    assert client.get("/v1/crud/model-templates").json() == []


def test_template_from_an_unreachable_service_is_a_bad_gateway(client, register_service, mock_wrapper_cls):
    mock_wrapper_cls.return_value.get_config_schema.side_effect = ConnectionError("service unreachable")
    register_service()
    response = client.post("/v1/crud/model-templates/from-service", json={"serviceId": "test-model"})
    assert response.status_code == 502
    assert client.get("/v1/crud/model-templates").json() == []


def test_non_chapkit_template_has_null_health_status(client, db_engine):
    from chap_core.database.database import SessionWrapper
    from chap_core.models.external_chapkit_model import ml_service_info_to_model_template_config

    info = MLServiceInfo.model_validate({**MOCK_INFO_DICT, "id": "non-chapkit-model"})
    config = ml_service_info_to_model_template_config(info, "http://localhost:9999")

    with Session(db_engine) as session:
        wrapper = SessionWrapper(session=session)
        wrapper.add_model_template_from_yaml_config(config)

    response = client.get("/v1/crud/model-templates")

    templates = response.json()
    matching = [t for t in templates if t["name"] == "non-chapkit-model"]
    assert len(matching) == 1
    assert matching[0]["healthStatus"] is None


def test_returns_200_when_redis_unavailable(client):
    with patch(
        "chap_core.rest_api.v2.dependencies.get_orchestrator",
        side_effect=ConnectionError("Redis unavailable"),
    ):
        response = client.get("/v1/crud/model-templates")

    assert response.status_code == 200
    assert response.json() == []


def test_failed_database_write_for_one_service_does_not_break_the_sync(client, register_service, db_engine):
    # A legacy unique display name constraint makes the template insert for test-model fail.
    with Session(db_engine) as session:
        session.execute(text("CREATE UNIQUE INDEX legacy_display_name_key ON modeltemplatedb (display_name)"))
        session.add(ModelTemplateDB(name="legacy-template", version="1.0.0", display_name="Test Model"))
        session.commit()
    register_service({**MOCK_INFO_DICT, "id": "other-model", "display_name": "Other Model"})
    register_service()

    response = client.get("/v1/crud/model-templates")

    assert response.status_code == 200
    names = {t["name"] for t in response.json()}
    assert "other-model" in names
    assert "test-model" not in names
