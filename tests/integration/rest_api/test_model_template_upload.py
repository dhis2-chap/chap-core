"""Integration tests for model template upload endpoint."""

import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from chap_core.rest_api.v1.rest_api import app

client = TestClient(app)


def create_zip_with_mlproject(mlproject_content: str) -> bytes:
    """Create a zip file with an MLproject file at root level."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("MLproject", mlproject_content)
    buffer.seek(0)
    return buffer.read()


def create_zip_without_mlproject() -> bytes:
    """Create a zip file without an MLproject file."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.md", "# Test model")
    buffer.seek(0)
    return buffer.read()


class TestUploadModelTemplate:
    def test_valid_zip_upload(self, dependency_overrides):
        mlproject_content = """name: test_model
entry_points:
  train:
    parameters:
      train_data: path
      model: str
    command: "python train.py {train_data} {model}"
  predict:
    parameters:
      future_data: path
      historic_data: path
      model: str
      out_file: path
    command: "python predict.py {future_data} {model} {out_file}"
"""
        zip_content = create_zip_with_mlproject(mlproject_content)
        files = {"zip_file": ("model.zip", zip_content, "application/zip")}

        response = client.post("/v1/crud/model-templates/upload", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_model"
        assert "id" in data
        assert data["sourceUrl"] == "upload://model.zip"

    def test_invalid_zip_file(self, dependency_overrides):
        files = {"zip_file": ("model.zip", b"not a zip file", "application/zip")}

        response = client.post("/v1/crud/model-templates/upload", files=files)

        assert response.status_code == 400
        assert "Invalid zip file" in response.json()["detail"]

    def test_zip_without_mlproject(self, dependency_overrides):
        zip_content = create_zip_without_mlproject()
        files = {"zip_file": ("model.zip", zip_content, "application/zip")}

        response = client.post("/v1/crud/model-templates/upload", files=files)

        assert response.status_code == 400
        assert "MLproject file not found" in response.json()["detail"]

    def test_invalid_yaml_in_mlproject(self, dependency_overrides):
        mlproject_content = """name: test_model
invalid yaml: [unclosed bracket
"""
        zip_content = create_zip_with_mlproject(mlproject_content)
        files = {"zip_file": ("model.zip", zip_content, "application/zip")}

        response = client.post("/v1/crud/model-templates/upload", files=files)

        assert response.status_code == 400
        assert "Invalid YAML" in response.json()["detail"]

    def test_invalid_mlproject_format(self, dependency_overrides):
        mlproject_content = """not_name: missing_required_field
"""
        zip_content = create_zip_with_mlproject(mlproject_content)
        files = {"zip_file": ("model.zip", zip_content, "application/zip")}

        response = client.post("/v1/crud/model-templates/upload", files=files)

        assert response.status_code == 400
        assert "Invalid MLproject format" in response.json()["detail"]
