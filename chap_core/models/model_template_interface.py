import abc


from chap_core.database.model_templates_and_config_tables import ModelTemplateInformation
from chap_core.models.configured_model import ModelConfiguration
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


class ConfiguredModel(abc.ABC):
    @abc.abstractmethod
    def train(self, train_data: DataSet, extra_args=None):
        pass

    @abc.abstractmethod
    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        pass


# class ModelConfiguration(BaseModel):
#     additional_continous_covariates: list[str] = []
#     user_options: dict = {}


class ModelTemplateInterface(abc.ABC):
    @abc.abstractmethod
    def get_schema(self) -> ModelTemplateInformation:
        return self.model_template_info

    @abc.abstractmethod
    def get_model(self, model_configuration: ModelConfiguration | None = None) -> "ConfiguredModel":
        pass

    def get_default_model(self) -> "ConfiguredModel":
        return self.get_model()


class InternalModelTemplate(ModelTemplateInterface):
    """
    This is a practical base class for defining model templates in python.
    The goal is that this can be used to define model templates that can be
    used directly in python, but also provide functionality for exposing them
    throught the chap/mlflow api
    """

    model_config_class: type[ModelConfiguration]
    model_template_info: ModelTemplateInformation

    def get_schema(self):
        return self.model_template_info.model_json_schema()
