"""Synchronous REST API wrapper for CHAPKit service."""

import logging
import time
from typing import Any, cast

import chapkit
import httpx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from chap_core.time_period.date_util_wrapper import pandas_period_to_string

logger = logging.getLogger(__name__)


class RunInfo(BaseModel):
    """Runtime information passed from CHAP to models."""

    prediction_length: int = Field(description="Number of periods to predict")
    additional_continuous_covariates: list[str] = Field(
        default_factory=list,
        description="User-specified additional covariates present in the data",
    )
    future_covariate_origin: str | None = Field(
        default=None,
        description="Origin/source of future covariate forecasts",
    )


def _prepare_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Convert a DataFrame to the columns/data format expected by CHAPKit."""
    if "time_period" in df.columns:
        df = df.copy()
        df["time_period"] = df["time_period"].apply(
            lambda x: pandas_period_to_string(x) if hasattr(x, "freqstr") else str(x)
        )
    df = df.replace({np.nan: None})
    return {"columns": df.columns.tolist(), "data": df.values.tolist()}


def _serialize_geo(geo_features: dict[str, Any] | None) -> dict[str, Any] | None:
    """Serialize geo features, converting Pydantic models to dicts."""
    if not geo_features:
        return None
    return geo_features if isinstance(geo_features, dict) else geo_features.model_dump()


class CHAPKitRestAPIWrapper:
    """Synchronous client for interacting with the CHAPKit REST API."""

    def __init__(self, base_url: str = "http://localhost:8001", timeout: int = 7200):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(
            base_url=self.base_url, timeout=self.timeout, headers={"Content-Type": "application/json"}
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _request(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        """Make synchronous HTTP request with error handling."""
        try:
            response = self.client.request(method, endpoint, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPError as e:
            raise httpx.HTTPError(f"API request failed: {e}") from e

    def close(self):
        """Close the client connection."""
        if self.client:
            self.client.close()

    # Information endpoints

    def health(self) -> dict[str, str]:
        """Check service health status."""
        response = self._request("GET", "/health")
        return cast("dict[str, str]", response.json())

    def info(self) -> dict[str, Any]:
        """Get system information."""
        response = self._request("GET", "/api/v1/info")
        return cast("dict[str, Any]", response.json())

    # Configuration management endpoints

    def list_configs(self) -> list[chapkit.ConfigOut]:
        """List all model configurations."""
        response = self._request("GET", "/api/v1/configs")
        return [chapkit.ConfigOut.model_validate(c) for c in response.json()]

    def create_config(self, config: dict[str, Any]) -> chapkit.ConfigOut:
        """Create or replace a model configuration."""
        response = self._request("POST", "/api/v1/configs", json=config)
        return chapkit.ConfigOut.model_validate(response.json())

    def get_config_schema(self) -> dict[str, Any]:
        """Get JSON Schema for model configuration."""
        response = self._request("GET", "/api/v1/configs/$schema")
        return cast("dict[str, Any]", response.json())

    # Job management endpoints

    def get_job(self, job_id: str) -> chapkit.ChapkitJobRecord:
        """Get full job record by ID."""
        response = self._request("GET", f"/api/v1/jobs/{job_id}")
        return chapkit.ChapkitJobRecord.model_validate(response.json())

    # Artifact endpoints

    def get_artifact(self, artifact_id: str) -> dict[str, Any]:
        """Get a specific artifact by ID."""
        response = self._request("GET", f"/api/v1/artifacts/{artifact_id}")
        return cast("dict[str, Any]", response.json())

    def get_prediction_artifact_dataframe(self, artifact_id: str) -> chapkit.data.DataFrame:
        """Get prediction artifact data as a chapkit DataFrame."""
        response = self.get_artifact(artifact_id)
        data = chapkit.artifact.schemas.MLPredictionArtifactData.model_validate(response["data"])
        return chapkit.data.DataFrame(**data.content)

    # CHAP operation endpoints

    def train(
        self,
        config_id: str,
        data: pd.DataFrame,
        run_info: RunInfo,
        geo_features: dict[str, Any] | None = None,
    ) -> chapkit.TrainResponse:
        """Train a model with data."""
        train_body: dict[str, Any] = {
            "config_id": config_id,
            "data": _prepare_dataframe(data),
            "run_info": run_info.model_dump(exclude_none=True),
        }

        geo = _serialize_geo(geo_features)
        if geo:
            train_body["geo"] = geo

        response = self._request("POST", "/api/v1/ml/$train", json=train_body)
        return chapkit.TrainResponse.model_validate(response.json())

    def predict(
        self,
        artifact_id: str,
        future_data: pd.DataFrame,
        run_info: RunInfo,
        historic_data: pd.DataFrame | None = None,
        geo_features: dict[str, Any] | None = None,
    ) -> chapkit.PredictResponse:
        """Make predictions with a trained model."""
        predict_body: dict[str, Any] = {
            "artifact_id": artifact_id,
            "future": _prepare_dataframe(future_data),
            "run_info": run_info.model_dump(exclude_none=True),
        }

        if historic_data is not None:
            predict_body["historic"] = _prepare_dataframe(historic_data)

        geo = _serialize_geo(geo_features)
        if geo:
            predict_body["geo"] = geo

        response = self._request("POST", "/api/v1/ml/$predict", json=predict_body)
        return chapkit.PredictResponse.model_validate(response.json())

    # Helper methods

    def wait_for_job(self, job_id: str, poll_interval: int = 2, timeout: int | None = None) -> chapkit.ChapkitJobRecord:
        """Wait for a job to complete."""
        start_time = time.time()

        while True:
            job = self.get_job(job_id)
            logger.info(f"Job {job_id} status: {job.status}")

            if job.status in ["completed", "failed", "canceled"]:
                return job

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")

            time.sleep(poll_interval)

    def train_and_wait(
        self,
        config_id: str,
        data: pd.DataFrame,
        run_info: RunInfo,
        geo_features: dict[str, Any] | None = None,
        timeout: int | None = 300,
    ) -> tuple[chapkit.ChapkitJobRecord, str]:
        """Train a model and wait for completion.

        Returns:
            Tuple of (job record, artifact_id)
        """
        result = self.train(config_id, data, run_info, geo_features)
        job = self.wait_for_job(result.job_id, timeout=timeout)
        return job, result.artifact_id

    def predict_and_wait(
        self,
        artifact_id: str,
        future_data: pd.DataFrame,
        run_info: RunInfo,
        historic_data: pd.DataFrame | None = None,
        geo_features: dict[str, Any] | None = None,
        timeout: int | None = 7200,
    ) -> tuple[chapkit.ChapkitJobRecord, str]:
        """Make predictions and wait for completion.

        Returns:
            Tuple of (job record, artifact_id)
        """
        result = self.predict(artifact_id, future_data, run_info, historic_data, geo_features)
        job = self.wait_for_job(result.job_id, timeout=timeout)
        return job, result.artifact_id
