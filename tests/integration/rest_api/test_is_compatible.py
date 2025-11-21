import logging

import pytest
from fastapi.testclient import TestClient

from chap_core.rest_api.v1.rest_api import app

client = TestClient(app, raise_server_exceptions=False)


def test_is_compatible_with_valid_version():
    """Test that a valid version returns the expected compatibility response"""
    response = client.get("/is-compatible?modelling_app_version=3.0.0")
    assert response.status_code == 200
    data = response.json()
    assert "compatible" in data
    assert "description" in data
    assert isinstance(data["compatible"], bool)


def test_is_compatible_with_invalid_version_format(caplog):
    """Test that an invalid version returns error without stack trace"""
    with caplog.at_level(logging.WARNING):
        response = client.get("/is-compatible?modelling_app_version=999.99.9-remove-orgunits-lacking-geometry.1")

    # Check response format
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert "error" in data
    assert "type" in data
    assert data["type"] == "InvalidVersion"
    assert "Invalid version" in data["error"]

    # Verify only WARNING level logs, no ERROR with stack trace
    assert any("Invalid version string" in record.message for record in caplog.records if record.levelname == "WARNING")
    # Verify no ERROR logs with "Full traceback" or traceback content
    assert not any("Full traceback" in record.message for record in caplog.records if record.levelname == "ERROR")


def test_is_compatible_with_old_version():
    """Test that an old version returns incompatible response"""
    response = client.get("/is-compatible?modelling_app_version=1.0.0")
    assert response.status_code == 200
    data = response.json()
    assert data["compatible"] is False
    assert "too old" in data["description"].lower()


def test_is_compatible_with_newer_version():
    """Test that a newer version returns compatible response"""
    response = client.get("/is-compatible?modelling_app_version=999.0.0")
    assert response.status_code == 200
    data = response.json()
    assert data["compatible"] is True
    assert "compatible" in data["description"].lower()
