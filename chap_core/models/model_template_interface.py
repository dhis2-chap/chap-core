import abc

from chap_core.external.model_configuration import ModelTemplateSchema
from chap_core.models.configured_model import ModelConfiguration
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


class ConfiguredModel(abc.ABC):
    @abc.abstractmethod
    def train(self, train_data: DataSet, extra_args=None):
        pass

    @abc.abstractmethod
    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        pass




class ModelTemplateInterface(abc.ABC):

    @abc.abstractmethod
    def get_config_class(self) -> type[ModelConfiguration]:  # gives a custom class of type ModelConfiguration
        # todo: could maybe be a property and not class
        pass

    @abc.abstractmethod
    def get_model(self, model_configuration: ModelConfiguration = None) -> 'ConfiguredModel':
        pass

    def get_default_model(self) -> 'ConfiguredModel':
        return self.get_model()


class InternalModelTemplate(ModelTemplateInterface):
    '''
    This is a practical base class for defining model templates in python.
    The goal is that this can be used to define model templates that can be
    used directly in python, but also provide functionality for exposing them
    throught the chap/mlflow api
    '''
    model_config_class: type[ModelConfiguration]
    required_fields: list[str] = []

    @property
    def model_template_info(self) -> ModelTemplateSchema:
        schema = self.model_config_class.model_json_schema()['properties']
        print(schema)
        ADDITIONAL_COVARIATE_NAME = 'additional_covariates'
        return ModelTemplateSchema(
            name=self.__class__.__name__,
            # description=self.model_config_class.__doc__,
            required_fields=[],
            allow_free_additional_continuous_covariates=schema.pop(ADDITIONAL_COVARIATE_NAME, None) is not None,
            user_options=schema
        )

    def get_config_class(self) -> type[ModelConfiguration]:
        return self.model_config_class

