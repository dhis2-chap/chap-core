"""Integration tests for model template CRUD endpoints."""

import logging
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from chap_core.database.model_templates_and_config_tables import ModelTemplateDB
from chap_core.models.approved_templates import ApprovedTemplate, clear_cache
from chap_core.rest_api.v1.rest_api import app

logger = logging.getLogger(__name__)
client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_template_cache():
    clear_cache()
    yield
    clear_cache()


class TestListAvailableModelTemplates:
    def test_returns_approved_list(self, dependency_overrides):
        mock_templates = [
            ApprovedTemplate(
                url="https://github.com/dhis2-chap/chap_auto_ewars",
                versions={"stable": "@abc123", "nightly": "@main"},
            ),
        ]

        with patch("chap_core.rest_api.v1.routers.crud.get_approved_templates") as mock_get:
            mock_get.return_value = mock_templates
            response = client.get("/v1/crud/model-templates/available")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["url"] == "https://github.com/dhis2-chap/chap_auto_ewars"
        assert "stable" in data[0]["versions"]

    def test_returns_empty_when_no_approved(self, dependency_overrides):
        with patch("chap_core.rest_api.v1.routers.crud.get_approved_templates") as mock_get:
            mock_get.return_value = []
            response = client.get("/v1/crud/model-templates/available")

        assert response.status_code == 200
        assert response.json() == []


class TestAddModelTemplate:
    @pytest.fixture
    def mock_approved_templates(self):
        return [
            ApprovedTemplate(
                url="https://github.com/dhis2-chap/chap_auto_ewars",
                versions={"stable": "@abc123"},
            ),
        ]

    def test_rejects_non_approved_url(self, dependency_overrides, mock_approved_templates):
        with patch("chap_core.rest_api.v1.routers.crud.get_approved_templates") as mock_get:
            mock_get.return_value = mock_approved_templates
            response = client.post(
                "/v1/crud/model-templates",
                json={"url": "https://github.com/evil/repo", "version": "@abc123"},
            )

        assert response.status_code == 403
        assert "not in approved list" in response.json()["detail"]

    def test_rejects_non_approved_version(self, dependency_overrides, mock_approved_templates):
        with patch("chap_core.rest_api.v1.routers.crud.get_approved_templates") as mock_get:
            mock_get.return_value = mock_approved_templates
            response = client.post(
                "/v1/crud/model-templates",
                json={
                    "url": "https://github.com/dhis2-chap/chap_auto_ewars",
                    "version": "@wrong_version",
                },
            )

        assert response.status_code == 403
        assert "not in approved list" in response.json()["detail"]


class TestDeleteModelTemplate:
    def test_delete_existing_template(self, override_session, p_seeded_engine):
        with Session(p_seeded_engine) as session:
            template = session.exec(select(ModelTemplateDB)).first()
            template_id = template.id

        response = client.delete(f"/v1/crud/model-templates/{template_id}")
        assert response.status_code == 200
        assert response.json()["message"] == "deleted"

        with Session(p_seeded_engine) as session:
            template = session.get(ModelTemplateDB, template_id)
            assert template is not None
            assert template.archived is True

    def test_delete_nonexistent_template(self, dependency_overrides):
        response = client.delete("/v1/crud/model-templates/99999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_archived_templates_not_listed(self, override_session, p_seeded_engine):
        with Session(p_seeded_engine) as session:
            template = session.exec(select(ModelTemplateDB)).first()
            template_id = template.id

        response = client.get("/v1/crud/model-templates")
        assert response.status_code == 200
        templates_before = response.json()
        ids_before = [t["id"] for t in templates_before]
        assert template_id in ids_before

        response = client.delete(f"/v1/crud/model-templates/{template_id}")
        assert response.status_code == 200

        response = client.get("/v1/crud/model-templates")
        assert response.status_code == 200
        templates_after = response.json()
        ids_after = [t["id"] for t in templates_after]
        assert template_id not in ids_after
