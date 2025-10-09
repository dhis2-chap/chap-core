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
        # always add a name, since chapkit needs that
        if "name" not in model_configuration:
            model_configuration["name"] = "default_name"

        config_response = self.client.create_config(model_configuration)
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

    def train(self, train_data: DataSet, extra_args=None):
        frequency = self._get_frequency(train_data)
        pd = train_data.to_pandas()
        pd["time_period"] = pd["time_period"].astype(str)
        new_pd = self._adapt_data(pd, frequency=frequency)
        geo = train_data.polygons
        job_id = self.client.train_and_wait(self.configuration_id, new_pd, geo)
        return job_id

    def predict(self, historic_data, future_data):
        pass
