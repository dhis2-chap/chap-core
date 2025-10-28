import logging
import pandas as pd
from chap_core.external.model_configuration import ModelTemplateConfigV2
from chap_core.models.external_model import ExternalModelBase
from chap_core.models.chapkit_rest_api_wrapper import CHAPKitRestAPIWrapper
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


from chap_core.datatypes import Samples

logger = logging.getLogger(__name__)


class ExternalChapkitModelTemplate:
    """Wrapper around External models that are based on chapkit.

    Note that get_model assumes you have already created a configuration with that specific chapkitmodel.

    This method is meant to be backwards compatible with ExternalModelTemplate
    """

    def __init__(self, rest_api_url: str):
        self.rest_api_url = rest_api_url
        self.client = CHAPKitRestAPIWrapper(rest_api_url)
        # assert self.is_healthy(), f"Service at {rest_api_url} is not healthy. Is model running? Check {self.rest_api_url}/health"

    def wait_for_healthy(self, timeout=60):
        import time

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_healthy():
                return True
            time.sleep(2)
        raise TimeoutError(
            f"Model service at {self.rest_api_url} did not become healthy within {timeout} seconds. Check {self.rest_api_url}/health"
        )

    def is_healthy(self) -> bool:
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
        info = self.client.info()
        if "name" in info:
            # name not supported in current chapkit version, might be supported in the future
            name = info["name"]
        else:
            name = info["display_name"].lower().replace(" ", "_")
        version = info.get("version")
        return f"{name}_v{version}"

    def get_model_template_config(self) -> ModelTemplateConfigV2:
        """
        This method is meant to make things backwards compatible with old system. An object of type
        ModelTemplateConfigV2 is needed to store info about a ModelTemplate in the database.
        """
        model_info = self.client.info()

        # Get user options from config schema
        config_schema = self.client.get_config_schema()
        print(config_schema)
        user_options = {}
        if "$defs" in config_schema and "ModelConfiguration" in config_schema["$defs"]:
            user_options = config_schema["$defs"]["ModelConfiguration"].get("properties", {})

        # Build metadata dict from info endpoint
        meta_data_dict = {
            "display_name": model_info.get("display_name", "No Display Name"),
            "description": model_info.get("description") or model_info.get("summary", "No Description"),
            "author_note": model_info.get("author_note", ""),
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
            "user_options": user_options,
            # ModelTemplateInformation fields will use defaults if not provided:
            # - supported_period_type defaults to PeriodType.any
            # - target defaults to "disease_cases"
            # RunnerConfig fields not needed for REST API models:
            "entry_points": None,
            "docker_env": None,
            "python_env": None,
            "source_url": self.rest_api_url,
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

    def train(self, train_data: DataSet, extra_args=None):
        frequency = self._get_frequency(train_data)
        pd = train_data.to_pandas()
        new_pd = self._adapt_data(pd, frequency=frequency)
        geo = train_data.polygons
        response = self.client.train_and_wait(self.configuration_id, new_pd, geo)

        if response["status"] == "failed":
            raise RuntimeError(f"Training failed: {response.get('error', 'Unknown error')}")

        artifact_id = response["model_artifact_id"]
        assert artifact_id is not None, response
        self._train_id = artifact_id
        return self

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        assert self._train_id is not None, "Model must be trained before prediction"
        geo = historic_data.polygons
        historic_data_pd = self._adapt_data(historic_data.to_pandas())
        future_data_pd = self._adapt_data(future_data.to_pandas())
        response = self.client.predict_and_wait(
            model_artifact_id=self._train_id,
            future_data=future_data_pd,
            historic_data=historic_data_pd,
            geo_features=geo,
        )

        if response["status"] == "failed":
            raise RuntimeError(f"Prediction failed: {response.get('error', 'Unknown error')}")

        artifact_id = response["prediction_artifact_id"]
        assert artifact_id is not None, response.get("error", "No prediction artifact")

        # get artifact from the client
        prediction = self.client.get_artifact(artifact_id)
        data = prediction["data"]["predictions"]
        return DataSet.from_pandas(pd.DataFrame(data=data["data"], columns=data["columns"]), Samples)
