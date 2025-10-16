import pandas as pd
from chap_core.models.external_model import ExternalModelBase
from chap_core.models.chapkit_rest_api_wrapper import CHAPKitRestAPIWrapper
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


class ExternalChapkitModelTemplate:
    """Wrapper around External models that are based on chapkit.

    Note that get_model assumes you have already created a configuration with that specific chapkitmodel.

    This method is meant to be backwards compatible with ExternalModelTemplate
    """

    def __init__(self, model_name: str, rest_api_url: str):
        self.model_name = model_name
        self.rest_api_url = rest_api_url
        self.client = CHAPKitRestAPIWrapper(rest_api_url)

    def get_model(self, model_configuration) -> "ExternalChapkitModel":
        """
        Sends the model configuration for storing in the model (by sending to the model rest api).
        This returns a configuration id back that we can use to identify the model.
        """
        import time

        # Create config with proper structure for new API
        # Use timestamp to make name unique
        timestamp = int(time.time() * 1000000)
        config_data = {
            "name": model_configuration.get("name", f"{self.model_name}_config_{timestamp}"),
            "data": model_configuration,
        }

        config_response = self.client.create_config(config_data)
        configuration_id = config_response["id"]
        return ExternalChapkitModel(self.model_name, self.rest_api_url, configuration_id=configuration_id)


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
        return artifact_id

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
        return DataSet.from_pandas(pd.DataFrame(data=data["data"], columns=data["columns"]))
