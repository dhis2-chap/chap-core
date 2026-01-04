import logging
from typing import Optional

from chap_core.datatypes import Samples
from chap_core.external.model_configuration import ModelTemplateConfigV2
from chap_core.model_spec import PeriodType
from chap_core.models.chapkit_rest_api_wrapper import CHAPKitRestAPIWrapper, RunInfo
from chap_core.models.chapkit_service_manager import ChapkitServiceManager, is_url
from chap_core.models.external_model import ExternalModelBase
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

logger = logging.getLogger(__name__)


class ExternalChapkitModelTemplate:
    """Wrapper around External models that are based on chapkit.

    Note that get_model assumes you have already created a configuration with that specific chapkitmodel.

    This class is meant to be backwards compatible with ExternalModelTemplate.

    Supports two modes:
    1. URL mode (backwards compatible): Pass a URL to a running service
    2. Directory mode (new): Pass a path to a model directory, service
       will be auto-started when used as context manager

    Example usage (directory mode):
        with ExternalChapkitModelTemplate("/path/to/model") as template:
            model = template.get_model({})
            model.train(data)
            predictions = model.predict(historic, future)

    Example usage (URL mode - backwards compatible):
        template = ExternalChapkitModelTemplate("http://localhost:8000")
        template.wait_for_healthy()
        model = template.get_model({})
    """

    def __init__(
        self,
        path_or_url: str,
        port: Optional[int] = None,
        host: str = "127.0.0.1",
        startup_timeout: int = 60,
    ):
        """
        Initialize the template.

        Args:
            path_or_url: Either a URL to a running service, or a path to
                        a model directory
            port: Port to use when auto-starting (directory mode only)
            host: Host to bind to when auto-starting (default: 127.0.0.1)
            startup_timeout: Seconds to wait for service startup
        """
        self._path_or_url = path_or_url
        self._port = port
        self._host = host
        self._startup_timeout = startup_timeout

        self._is_url_mode = is_url(path_or_url)

        if self._is_url_mode:
            self.rest_api_url = path_or_url
            self.client = CHAPKitRestAPIWrapper(path_or_url)
            self._service_manager: Optional[ChapkitServiceManager] = None
        else:
            self.rest_api_url = None
            self.client = None
            self._service_manager = ChapkitServiceManager(
                model_directory=path_or_url,
                port=port,
                host=host,
                startup_timeout=startup_timeout,
            )

    def _ensure_initialized(self) -> None:
        """Ensure the template is initialized (service running, client ready)."""
        if self.client is None:
            raise RuntimeError(
                "Template not initialized. When using directory mode, "
                "use as context manager: with ExternalChapkitModelTemplate(path) as template: ..."
            )

    def __enter__(self) -> "ExternalChapkitModelTemplate":
        """Start service if in directory mode."""
        if self._is_url_mode:
            return self

        self._service_manager.__enter__()
        self.rest_api_url = self._service_manager.url
        self.client = CHAPKitRestAPIWrapper(self.rest_api_url)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop service if in directory mode."""
        if self._service_manager is not None and not self._is_url_mode:
            self._service_manager.__exit__(exc_type, exc_val, exc_tb)
            self.rest_api_url = None
            self.client = None

    def wait_for_healthy(self, timeout=60):
        self._ensure_initialized()
        import time

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_healthy():
                return True
            logger.info("Waiting for model service to become healthy...")
            time.sleep(2)
        raise TimeoutError(
            f"Model service at {self.rest_api_url} did not become healthy within {timeout} seconds. Check {self.rest_api_url}/health"
        )

    def is_healthy(self) -> bool:
        self._ensure_initialized()
        try:
            response = self.client.health()
            return response["status"] == "healthy"
        except Exception as e:
            logger.info(
                f"Health check for model {self.rest_api_url} failed: {e}. Check health at {self.rest_api_url}/health"
            )
            return False

    def get_model(self, model_configuration: dict) -> "ExternalChapkitModel":
        """
        Sends the model configuration for storing in the model (by sending to the model rest api).
        This returns a configuration id back that we can use to identify the model.
        """
        self._ensure_initialized()
        import time

        if model_configuration is None:
            model_configuration = {}
        else:
            model_configuration = dict(model_configuration)

        timestamp = int(time.time() * 1000000)
        if "name" not in model_configuration:
            name = f"{self.name}_config_{timestamp}"
        else:
            # always make sure config has unique name for now. Chapkit uses name as identifier,
            # but we don't necesserarily do that on the chap side
            name = model_configuration["name"] + "_" + str(timestamp)

        if "model_template" in model_configuration:
            # remove model_template key
            model_configuration.pop("model_template")

        config_data = {"name": name, "data": model_configuration}
        logger.info(f"Creating model configuration with name {name} at {self.rest_api_url}. Data: {config_data}")

        # Create config with proper structure for new API
        # Use timestamp to make name unique
        # config_data = {
        #    "name": model_configuration.get("name", f"{self.name}_config_{timestamp}"),
        #    "data": model_configuration
        # }

        config_response = self.client.create_config(config_data)
        configuration_id = config_response["id"]

        # get all configs and assert that configuration_id is there
        all_configs = self.client.list_configs()
        assert any(cfg["id"] == configuration_id for cfg in all_configs), (
            f"Created configuration {configuration_id} not found in list of configs"
        )

        logger.info(f"Created model configuration with id {configuration_id} at {self.rest_api_url}")
        return ExternalChapkitModel(self.name, self.rest_api_url, configuration_id=configuration_id)

    @property
    def name(self):
        """
        This returns a unique name for the model. In the future, this might be some sort of id given by the model
        """
        self._ensure_initialized()
        info = self.client.info()
        if "name" in info:
            # name not supported in current chapkit version, might be supported in the future
            name = info["name"]
        else:
            name = info["display_name"].lower().replace(" ", "_")
        version = info.get("version")
        return f"{name}_v{version}"

    @property
    def model_template_config(self) -> ModelTemplateConfigV2:
        """Property alias for get_model_template_config for backwards compatibility."""
        return self.get_model_template_config()

    def get_model_template_config(self) -> ModelTemplateConfigV2:
        """
        This method is meant to make things backwards compatible with old system. An object of type
        ModelTemplateConfigV2 is needed to store info about a ModelTemplate in the database.
        """
        self._ensure_initialized()
        model_info = self.client.info()

        # Get user options from config schema
        config_schema = self.client.get_config_schema()
        user_options = {}
        if "$defs" in config_schema and "ModelConfiguration" in config_schema["$defs"]:
            user_options = config_schema["$defs"]["ModelConfiguration"].get("properties", {})

        # Build metadata dict from info endpoint
        # TODO: this can be done by dumping the dict into a base-model
        meta_data_dict = {
            "display_name": model_info.get("display_name", "No Display Name"),
            "description": model_info.get("description") or model_info.get("summary", "No Description"),
            "author_note": model_info.get("author_note") or "",
            "author_assessed_status": model_info.get("author_assessed_status", "red"),
            "author": model_info.get("author", "Unknown Author"),
            "organization": model_info.get("organization"),
            "organization_logo_url": model_info.get("organization_logo_url"),
            "contact_email": model_info.get("contact_email"),
            "citation_info": model_info.get("citation_info"),
        }

        # Build complete config dict
        config_dict = {
            "name": self.name,
            "rest_api_url": self.rest_api_url,
            "meta_data": meta_data_dict,
            "required_covariates": model_info.get("required_covariates", []),
            "allow_free_additional_continuous_covariates": model_info.get(
                "allow_free_additional_continuous_covariates", False
            ),
            "supported_period_type": PeriodType(model_info.get("supported_period_type", "any")),
            "user_options": user_options,
            # target defaults to "disease_cases"
            # RunnerConfig fields not needed for REST API models:
            "entry_points": None,
            "docker_env": None,
            "python_env": None,
            "source_url": self._path_or_url,
        }

        return ModelTemplateConfigV2.model_validate(config_dict)


class ExternalChapkitModel(ExternalModelBase):
    def __init__(self, model_name: str, rest_api_url: str, configuration_id: str):
        self.model_name = model_name
        self.rest_api_url = rest_api_url
        self.configuration_id = configuration_id
        self._location_mapping = None
        self._adapters = None
        self.client = CHAPKitRestAPIWrapper(rest_api_url)
        self._train_id = None

    def train(self, train_data: DataSet, extra_args=None, run_info: RunInfo | None = None):
        frequency = self._get_frequency(train_data)
        df = train_data.to_pandas()
        new_df = self._adapt_data(df, frequency=frequency)
        geo = train_data.polygons
        if run_info is None:
            run_info = RunInfo(prediction_length=1)
        response = self.client.train_and_wait(self.configuration_id, new_df, run_info, geo)

        if response["status"] == "failed":
            raise RuntimeError(
                f"Training failed: {response.get('error', 'Unknown error')}. Stacktrace: {response.get('error_traceback', '')}"
            )

        artifact_id = response["artifact_id"]
        assert artifact_id is not None, response
        self._train_id = artifact_id
        return self

    def predict(self, historic_data: DataSet, future_data: DataSet, run_info: RunInfo | None = None) -> DataSet:
        assert self._train_id is not None, "Model must be trained before prediction"
        geo = historic_data.polygons
        historic_data_pd = self._adapt_data(historic_data.to_pandas())
        future_data_pd = self._adapt_data(future_data.to_pandas())
        if run_info is None:
            prediction_length = len(future_data.period_range)
            run_info = RunInfo(prediction_length=prediction_length)
        response = self.client.predict_and_wait(
            artifact_id=self._train_id,
            future_data=future_data_pd,
            run_info=run_info,
            historic_data=historic_data_pd,
            geo_features=geo,
        )

        if response["status"] == "failed":
            raise RuntimeError(f"Prediction failed: {response.get('error', 'Unknown error')}")

        artifact_id = response["artifact_id"]
        assert artifact_id is not None, response.get("error", "No prediction artifact")

        # get artifact from the client
        prediction_data = self.client.get_prediction_artifact_dataframe(artifact_id)
        pd = prediction_data.to_pandas()
        # Sort by location and time_period to ensure consecutive periods
        if "time_period" in pd.columns and "location" in pd.columns:
            pd = pd.sort_values(by=["location", "time_period"]).reset_index(drop=True)
        return DataSet.from_pandas(pd, Samples)
