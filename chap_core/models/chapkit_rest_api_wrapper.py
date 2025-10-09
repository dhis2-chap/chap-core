"""
Synchronous REST API wrapper for CHAP service
Provides synchronous methods for all available API endpoints

NOTE: Written by ai as a prototype, TODO: refactor and cleanup once working

"""

import logging
import time
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import httpx

logger = logging.getLogger(__name__)


class CHAPKitRestAPIWrapper:
    """Synchronous client for interacting with the CHAP REST API"""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 7200):
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
            Dict with status field ('up' or 'down')
        """
        response = self._request("GET", "/api/v1/health")
        return response.json()

    def info(self) -> Dict[str, Any]:
        """
        Get service information

        Returns:
            Service info including display_name, author, description, etc.
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
        response = self._request("GET", "/api/v1/configs/schema")
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

    def get_artifact_tree(self, config_id: str) -> List[Dict[str, Any]]:
        """
        Get artifacts for a configuration as a tree structure

        Args:
            config_id: Configuration ID

        Returns:
            Artifact tree with nested children
        """
        params = {"config_id": config_id}
        response = self._request("GET", "/api/v1/artifacts/tree", params=params)
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

    def delete_artifact(self, artifact_id: str) -> None:
        """
        Delete an artifact by ID

        Args:
            artifact_id: Artifact ID to delete
        """
        self._request("DELETE", f"/api/v1/artifacts/{artifact_id}")

    # CHAP operation endpoints

    def train(self, config_id: str, data: pd.DataFrame, geo_features: Optional[Dict[str, Any]] = None) -> str:
        """
        Train a model with data

        Args:
            config_id: Configuration ID to use for training
            data: Training data as pandas DataFrame
            geo_features: Optional GeoJSON FeatureCollection

        Returns:
            Job ID for the training task
        """
        # Convert DataFrame to split format
        # data = data.fillna(None)  # so that it can be encoded to json
        data["time_period"] = data["time_period"].astype(str)
        data = data.replace({np.nan: None})
        train_body = {
            "data": data.to_dict(orient="split")
            # "columns": data.columns.tolist(),
            # "index": data.index.tolist() if not data.index.equals(pd.RangeIndex(len(data))) else None,
            # "data": data.values.tolist()
        }

        if geo_features:
            train_body["geo"] = geo_features

        params = {"config": config_id}
        response = self._request("POST", "/api/v1/train", params=params, json=train_body)
        return response.json()["id"]

    def predict(
        self,
        artifact_id: str,
        historic_data: pd.DataFrame,
        future_data: pd.DataFrame,
        geo_features: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Make predictions with a trained model

        Args:
            artifact_id: Trained artifact ID
            historic_data: Historical data as pandas DataFrame
            future_data: Future covariates as pandas DataFrame
            geo_features: Optional GeoJSON FeatureCollection

        Returns:
            Job ID for the prediction task
        """
        historic_data["time_period"] = historic_data["time_period"].astype(str)
        historic_data = historic_data.replace({np.nan: None})
        future_data["time_period"] = future_data["time_period"].astype(str)
        future_data = future_data.replace({np.nan: None})

        predict_body = {
            "historic": historic_data.to_dict(orient="split"),
            "future": future_data.to_dict(orient="split"),
        }

        if geo_features:
            predict_body["geo"] = geo_features

        params = {"artifact": artifact_id}
        response = self._request("POST", "/api/v1/predict", params=params, json=predict_body)
        return response.json()["id"]

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
        geo_features: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = 300,
    ) -> Dict[str, Any]:
        """
        Train a model and wait for completion

        Args:
            config_id: Configuration ID
            data: Training data
            geo_features: Optional GeoJSON features
            timeout: Maximum seconds to wait

        Returns:
            Completed job record with artifact_id
        """
        job_id = self.train(config_id, data, geo_features)
        return self.wait_for_job(job_id, timeout=timeout)

    def predict_and_wait(
        self,
        artifact_id: str,
        historic_data: pd.DataFrame,
        future_data: pd.DataFrame,
        geo_features: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = 7200,
    ) -> Dict[str, Any]:
        """
        Make predictions and wait for completion

        Args:
            artifact_id: Trained artifact ID
            historic_data: Historical data
            future_data: Future covariates
            geo_features: Optional GeoJSON features
            timeout: Maximum seconds to wait

        Returns:
            Completed job record with results
        """
        job_id = self.predict(artifact_id, historic_data, future_data, geo_features)
        return self.wait_for_job(job_id, timeout=timeout)

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
