import abc
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