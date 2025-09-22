import io
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yaml

from chap_core.datatypes import Samples
from chap_core.exceptions import ModelFailedException
from chap_core.geometry import Polygons
from chap_core.models.external_model import ExternalModelBase
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

logger = logging.getLogger(__name__)


class ExternalWebModel(ExternalModelBase):
    """
    Wrapper for a ConfiguredModel that can only be run through a web service, defined by an URL.
    This web service supports a strict REST API that allows for training and prediction.
    This class makes such a model available through the ConfiguredModel interface,
    with train and predict methods.
    """

    def __init__(
        self,
        api_url: str,
        name: str = None,
        timeout: int = 3600,
        poll_interval: int = 5,
        configuration: Optional[dict] = None,
        adapters: Optional[dict] = None,
        working_dir: str = "./",
    ):
        """
        Initialize the ExternalWebModel.

        Parameters
        ----------
        api_url : str
            Base URL of the model API (e.g., "http://localhost:8000")
        name : str
            Name of the model
        timeout : int
            Maximum time to wait for job completion in seconds (default: 3600)
        poll_interval : int
            Time between status checks in seconds (default: 5)
        configuration : dict
            Optional configuration to pass to the model
        adapters : dict
            Optional field name adapters for data conversion
        working_dir : str
            Working directory for temporary files (default: "./")
        """
        self._api_url = api_url.rstrip("/")
        self._name = name or "external_web_model"
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._configuration = configuration or {}
        self._trained_model_name = None
        self._adapters = adapters
        self._working_dir = Path(working_dir)
        self._location_mapping = None
        self._polygons_file_name = None

    @property
    def name(self):
        return self._name

    @property
    def configuration(self):
        return self._configuration

    def _wait_for_job(self, job_id: str, job_type: str = "job") -> dict:
        """
        Poll the API until the job completes or times out.

        Parameters
        ----------
        job_id : str
            The job ID to monitor
        job_type : str
            Type of job for logging ("training" or "prediction")

        Returns
        -------
        dict
            The final job status

        Raises
        ------
        ModelFailedException
            If the job fails or times out
        """
        start_time = time.time()

        while time.time() - start_time < self._timeout:
            try:
                response = requests.get(f"{self._api_url}/check_status/{job_id}")
                response.raise_for_status()
                job_info = response.json()

                status = job_info["status"]
                logger.info(f"{job_type} job {job_id} status: {status}")

                if status == "completed":
                    return job_info
                elif status == "failed":
                    error_msg = job_info.get("error_message", "Unknown error")
                    raise ModelFailedException(f"{job_type} job failed: {error_msg}")

                time.sleep(self._poll_interval)

            except requests.RequestException as e:
                logger.error(f"Error checking job status: {e}")
                raise ModelFailedException(f"Failed to check job status: {e}")

        raise ModelFailedException(f"{job_type} job timed out after {self._timeout} seconds")

    def train(self, train_data: DataSet, extra_args=None):
        """
        Trains the model by starting training and waiting for the model to finish training
        """
        logger.info(f"Training {self._name} via API at {self._api_url}")

        # Adapt training data using base class method
        frequency = self._get_frequency(train_data)
        train_df = train_data.to_pandas()
        adapted_df = self._adapt_data(train_df, frequency=frequency)

        # Convert to CSV string for upload
        train_csv = io.StringIO()
        adapted_df.to_csv(train_csv, index=False)
        train_csv.seek(0)

        # Prepare files for upload
        files = {
            "training_data": ("training_data.csv", train_csv, "text/csv"),
        }

        # Handle polygons using base class method
        if train_data.polygons is not None:
            self._polygons_file_name = self._working_dir / "polygons.geojson"
            self._write_polygons_to_geojson(train_data, self._polygons_file_name)
            polygons_json = Polygons(train_data.polygons).to_json()
            files["polygons"] = ("polygons.geojson", polygons_json, "application/geo+json")
            logger.info("Will pass polygons file to train and predict commands")

        # Add configuration if present
        if self._configuration:
            config_yaml = yaml.dump(self._configuration)
            files["config"] = ("config.yaml", config_yaml, "text/yaml")

        # Generate unique model name for this training session
        self._trained_model_name = f"{self._name}_{uuid.uuid4().hex[:8]}"

        data = {
            "model_name": self._trained_model_name,
        }

        try:
            # Submit training job
            response = requests.post(
                f"{self._api_url}/train",
                files=files,
                data=data,
            )
            response.raise_for_status()

            result = response.json()
            job_id = result["job_id"]
            logger.info(f"Training job submitted with ID: {job_id}")

            # Wait for training to complete
            self._wait_for_job(job_id, "Training")
            logger.info(f"Training completed for model: {self._trained_model_name}")

            return self

        except requests.RequestException as e:
            logger.error(f"Failed to submit training job: {e}")
            raise ModelFailedException(f"Training failed: {e}")

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        """
        Predicts by starting prediction and waiting for the model to finish prediction.
        """
        if self._trained_model_name is None:
            raise ModelFailedException("Model must be trained before prediction")

        logger.info(f"Predicting with {self._trained_model_name} via API at {self._api_url}")

        # Adapt data using base class methods
        historic_frequency = self._get_frequency(historic_data)
        future_frequency = self._get_frequency(future_data)

        historic_adapted = self._adapt_data(historic_data.to_pandas(), frequency=historic_frequency)
        future_adapted = self._adapt_data(future_data.to_pandas(), frequency=future_frequency)

        # Convert adapted data to CSV strings
        historic_csv = io.StringIO()
        historic_adapted.to_csv(historic_csv, index=False)
        historic_csv.seek(0)

        future_csv = io.StringIO()
        future_adapted.to_csv(future_csv, index=False)
        future_csv.seek(0)

        # Prepare files for upload
        files = {
            "historic_data": ("historic_data.csv", historic_csv, "text/csv"),
            "future_data": ("future_data.csv", future_csv, "text/csv"),
        }

        # Add polygons if present (reuse from training if available)
        if self._polygons_file_name is not None:
            # Use existing polygons from training
            polygons_json = Polygons(future_data.polygons).to_json()
            files["polygons"] = ("polygons.geojson", polygons_json, "application/geo+json")
        elif future_data.polygons is not None:
            # Write new polygons if not already written during training
            self._polygons_file_name = self._working_dir / "polygons.geojson"
            self._write_polygons_to_geojson(future_data, self._polygons_file_name)
            polygons_json = Polygons(future_data.polygons).to_json()
            files["polygons"] = ("polygons.geojson", polygons_json, "application/geo+json")

        data = {
            "model_name": self._trained_model_name,
        }

        try:
            # Submit prediction job
            response = requests.post(
                f"{self._api_url}/predict",
                files=files,
                data=data,
            )
            response.raise_for_status()

            result = response.json()
            job_id = result["job_id"]
            logger.info(f"Prediction job submitted with ID: {job_id}")

            # Wait for prediction to complete
            self._wait_for_job(job_id, "Prediction")

            # Fetch predictions
            response = requests.get(f"{self._api_url}/fetch_predictions/{job_id}")
            response.raise_for_status()

            # Parse predictions CSV
            predictions_csv = io.StringIO(response.text)
            predictions_df = pd.read_csv(predictions_csv)

            # Apply inverse location mapping if needed
            if self._location_mapping is not None:
                predictions_df["location"] = predictions_df["location"].apply(self._location_mapping.index_to_name)

            # Convert to DataSet
            try:
                predictions = DataSet.from_pandas(predictions_df, Samples)
                return predictions
            except ValueError as e:
                logger.error(f"Error parsing predictions: {e}")
                raise ModelFailedException(f"Failed to parse predictions: {e}")

        except requests.RequestException as e:
            logger.error(f"Failed to submit prediction job: {e}")
            raise ModelFailedException(f"Prediction failed: {e}")
