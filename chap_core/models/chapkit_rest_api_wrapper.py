"""
Synchronous REST API wrapper for CHAP service
Provides synchronous methods for all available API endpoints

NOTE: Written by ai as a prototype, TODO: refactor and cleanup once working

"""

import logging
import chapkit
import time
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import httpx
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


class CHAPKitRestAPIWrapper:
    """Synchronous client for interacting with the CHAP REST API"""

    def __init__(self, base_url: str = "http://localhost:8001", timeout: int = 7200):
        """
        Initialize the API client

        Args:
            base_url: Base URL of the CHAP API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(
            base_url=self.base_url, timeout=self.timeout, headers={"Content-Type": "application/json"}
        )

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def _request(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        """Make synchronous HTTP request with error handling"""
        try:
            response = self.client.request(method, endpoint, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPError as e:
            raise httpx.HTTPError(f"API request failed: {e}") from e

    def close(self):
        """Close the client connection"""
        if self.client:
            self.client.close()

    # Information endpoints

    def health(self) -> Dict[str, str]:
        """
        Check service health status

        Returns:
            Dict with status field ('healthy')
        """
        response = self._request("GET", "/health")
        return response.json()

    def info(self) -> Dict[str, Any]:
        """
        Get system information

        Returns:
            System info including name, version, description, etc.
        """
        response = self._request("GET", "/api/v1/info")
        return response.json()

    # Configuration management endpoints

    def list_configs(self) -> List[Dict[str, Any]]:
        """
        List all model configurations

        Returns:
            List of model configuration objects
        """
        response = self._request("GET", "/api/v1/configs")
        return response.json()

    def create_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create or replace a model configuration

        Args:
            config: Configuration dictionary

        Returns:
            Created configuration with ID
        """
        response = self._request("POST", "/api/v1/configs", json=config)
        return response.json()

    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get JSON Schema for model configuration

        Returns:
            JSON Schema for configuration model
        """
        response = self._request("GET", "/api/v1/configs/$schema")
        return response.json()

    def get_config(self, config_id: str) -> Dict[str, Any]:
        """
        Get a specific configuration by ID

        Args:
            config_id: Configuration ID

        Returns:
            Configuration object
        """
        response = self._request("GET", f"/api/v1/configs/{config_id}")
        return response.json()

    def update_config(self, config_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a configuration by ID

        Args:
            config_id: Configuration ID
            config: Updated configuration dictionary

        Returns:
            Updated configuration
        """
        response = self._request("PUT", f"/api/v1/configs/{config_id}", json=config)
        return response.json()

    def delete_config(self, config_id: str) -> None:
        """
        Delete a configuration by ID

        Args:
            config_id: Configuration ID to delete
        """
        self._request("DELETE", f"/api/v1/configs/{config_id}")

    def link_artifact_to_config(self, config_id: str, artifact_id: str) -> Dict[str, Any]:
        """
        Link an artifact to a configuration

        Args:
            config_id: Configuration ID
            artifact_id: Artifact ID to link

        Returns:
            Updated configuration or confirmation
        """
        response = self._request(
            "POST", f"/api/v1/configs/{config_id}/$link-artifact", json={"artifact_id": artifact_id}
        )
        return response.json()

    def unlink_artifact_from_config(self, config_id: str, artifact_id: str) -> Dict[str, Any]:
        """
        Unlink an artifact from a configuration

        Args:
            config_id: Configuration ID
            artifact_id: Artifact ID to unlink

        Returns:
            Updated configuration or confirmation
        """
        response = self._request(
            "POST", f"/api/v1/configs/{config_id}/$unlink-artifact", json={"artifact_id": artifact_id}
        )
        return response.json()

    def get_config_artifacts(self, config_id: str) -> List[Dict[str, Any]]:
        """
        Get all artifacts linked to a configuration

        Args:
            config_id: Configuration ID

        Returns:
            List of artifact objects linked to the configuration
        """
        response = self._request("GET", f"/api/v1/configs/{config_id}/$artifacts")
        return response.json()

    # Job management endpoints

    def get_jobs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all jobs, optionally filtered by status

        Args:
            status: Optional status filter ('pending', 'running', 'completed', 'failed', 'canceled')

        Returns:
            List of job records
        """
        params = {"status": status} if status else {}
        response = self._request("GET", "/api/v1/jobs", params=params)
        return response.json()

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """
        Get full job record by ID

        Args:
            job_id: Job ID

        Returns:
            Job record with status, times, error info, etc.
        """
        response = self._request("GET", f"/api/v1/jobs/{job_id}")
        return response.json()

    def delete_job(self, job_id: str) -> None:
        """
        Cancel (if running) and delete a job

        Args:
            job_id: Job ID to delete
        """
        self._request("DELETE", f"/api/v1/jobs/{job_id}")

    # Artifact management endpoints

    def get_artifacts_for_config(self, config_id: str) -> List[Dict[str, Any]]:
        """
        Get all artifacts linked to a configuration

        Args:
            config_id: Configuration ID

        Returns:
            List of artifact info objects
        """
        response = self._request("GET", f"/api/v1/artifacts/config/{config_id}")
        return response.json()

    def get_artifact(self, artifact_id: str) -> Dict[str, Any]:
        """
        Get a specific artifact by ID

        Args:
            artifact_id: Artifact ID

        Returns:
            Artifact info object
        """
        response = self._request("GET", f"/api/v1/artifacts/{artifact_id}")
        return response.json()

    def get_prediction_artifact_dataframe(self, artifact_id: str) -> chapkit.data.DataFrame:
        """
        Get the data content of a specific artifact by ID

        Args:
            artifact_id: Artifact ID

        Returns:
            Artifact data object
        """
        response = self.get_artifact(artifact_id)
        # note: this is a dict of a chapkit DataFrame
        data = chapkit.artifact.schemas.MLPredictionArtifactData.model_validate(response["data"])
        return chapkit.data.DataFrame(**data.content)

    def delete_artifact(self, artifact_id: str) -> None:
        """
        Delete an artifact by ID

        Args:
            artifact_id: Artifact ID to delete
        """
        self._request("DELETE", f"/api/v1/artifacts/{artifact_id}")

    def get_artifact_expand(self, artifact_id: str) -> Dict[str, Any]:
        """
        Get artifact with expanded data

        Args:
            artifact_id: Artifact ID

        Returns:
            Expanded artifact object
        """
        response = self._request("GET", f"/api/v1/artifacts/{artifact_id}/$expand")
        return response.json()

    def get_artifact_tree_by_id(self, artifact_id: str) -> Dict[str, Any]:
        """
        Get artifact tree starting from a specific artifact

        Args:
            artifact_id: Artifact ID

        Returns:
            Artifact tree with nested children
        """
        response = self._request("GET", f"/api/v1/artifacts/{artifact_id}/$tree")
        return response.json()

    def get_artifact_config(self, artifact_id: str) -> Dict[str, Any]:
        """
        Get the configuration associated with an artifact

        Args:
            artifact_id: Artifact ID

        Returns:
            Configuration object linked to the artifact
        """
        response = self._request("GET", f"/api/v1/artifacts/{artifact_id}/$config")
        return response.json()

    # CHAP operation endpoints

    def train(
        self,
        config_id: str,
        data: pd.DataFrame,
        run_info: RunInfo,
        geo_features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Train a model with data

        Args:
            config_id: Configuration ID to use for training
            data: Training data as pandas DataFrame
            run_info: Runtime information for the model
            geo_features: Optional GeoJSON FeatureCollection

        Returns:
            Dict with job_id and artifact_id
        """
        # Convert DataFrame to split format
        if "time_period" in data.columns:
            data["time_period"] = data["time_period"].apply(
                lambda x: pandas_period_to_string(x) if hasattr(x, "freqstr") else str(x)
            )
        data = data.replace({np.nan: None})

        # Convert DataFrame to columns/data format
        train_body: Dict[str, Any] = {
            "config_id": config_id,
            "data": {"columns": data.columns.tolist(), "data": data.values.tolist()},
            "run_info": run_info.model_dump(exclude_none=True),
        }

        if geo_features:
            train_body["geo"] = geo_features

        response = self._request("POST", "/api/v1/ml/$train", json=train_body)
        result = response.json()
        return {"job_id": result["job_id"], "artifact_id": result["artifact_id"]}

    def predict(
        self,
        artifact_id: str,
        future_data: pd.DataFrame,
        run_info: RunInfo,
        historic_data: Optional[pd.DataFrame] = None,
        geo_features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Make predictions with a trained model

        Args:
            artifact_id: Trained model artifact ID
            future_data: Future covariates as pandas DataFrame
            run_info: Runtime information for the model
            historic_data: Optional historical data as pandas DataFrame
            geo_features: Optional GeoJSON FeatureCollection

        Returns:
            Dict with job_id and artifact_id
        """
        if "time_period" in future_data.columns:
            future_data["time_period"] = future_data["time_period"].apply(
                lambda x: pandas_period_to_string(x) if hasattr(x, "freqstr") else str(x)
            )
        future_data = future_data.replace({np.nan: None})

        predict_body: Dict[str, Any] = {
            "artifact_id": artifact_id,
            "future": {"columns": future_data.columns.tolist(), "data": future_data.values.tolist()},
            "run_info": run_info.model_dump(exclude_none=True),
        }

        if historic_data is not None:
            if "time_period" in historic_data.columns:
                historic_data["time_period"] = historic_data["time_period"].apply(
                    lambda x: pandas_period_to_string(x) if hasattr(x, "freqstr") else str(x)
                )
            historic_data = historic_data.replace({np.nan: None})
            predict_body["historic"] = {
                "columns": historic_data.columns.tolist(),
                "data": historic_data.values.tolist(),
            }

        if geo_features:
            predict_body["geo"] = geo_features

        response = self._request("POST", "/api/v1/ml/$predict", json=predict_body)
        result = response.json()
        return {"job_id": result["job_id"], "artifact_id": result["artifact_id"]}

    # Helper methods

    def wait_for_job(self, job_id: str, poll_interval: int = 2, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Wait for a job to complete

        Args:
            job_id: Job ID to monitor
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait (None for no timeout)

        Returns:
            Final job status

        Raises:
            TimeoutError: If job doesn't complete within timeout
        """
        start_time = time.time()

        while True:
            job = self.get_job(job_id)
            logger.info(f"Job {job_id} status: {job['status']}")

            if job["status"] in ["completed", "failed", "canceled"]:
                return job

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")

            time.sleep(poll_interval)

    def train_and_wait(
        self,
        config_id: str,
        data: pd.DataFrame,
        run_info: RunInfo,
        geo_features: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = 300,
    ) -> Dict[str, Any]:
        """
        Train a model and wait for completion

        Args:
            config_id: Configuration ID
            data: Training data
            run_info: Runtime information for the model
            geo_features: Optional GeoJSON features
            timeout: Maximum seconds to wait

        Returns:
            Dict with job record and artifact_id
        """
        result = self.train(config_id, data, run_info, geo_features)
        job_id = result["job_id"]
        job = self.wait_for_job(job_id, timeout=timeout)
        job["artifact_id"] = result["artifact_id"]
        return job

    def predict_and_wait(
        self,
        artifact_id: str,
        future_data: pd.DataFrame,
        run_info: RunInfo,
        historic_data: Optional[pd.DataFrame] = None,
        geo_features: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = 7200,
    ) -> Dict[str, Any]:
        """
        Make predictions and wait for completion

        Args:
            artifact_id: Trained model artifact ID
            future_data: Future covariates
            run_info: Runtime information for the model
            historic_data: Optional historical data
            geo_features: Optional GeoJSON features
            timeout: Maximum seconds to wait

        Returns:
            Dict with job record and artifact_id
        """
        result = self.predict(artifact_id, future_data, run_info, historic_data, geo_features)
        job_id = result["job_id"]
        job = self.wait_for_job(job_id, timeout=timeout)
        job["artifact_id"] = result["artifact_id"]
        return job

    def poll_job(self, job_id: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Simple polling method that waits for a job to complete

        Args:
            job_id: Job ID to poll
            timeout: Maximum seconds to wait (None for no timeout)

        Returns:
            Final job status when completed
        """
        return self.wait_for_job(job_id, timeout=timeout)
