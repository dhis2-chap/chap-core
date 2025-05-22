from pydantic import BaseModel
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
import abc


class ConfiguredModel(abc.ABC):
    """
    A ConfiguredModel is the main interface for all models in the Chap framework.
    A configured model is different from a model template in that it is configured with specific hyperparameters
    and/or other choices. While a ModelTemplate is flexible with choices, a ConfiguredModel has fixed choices
    and parameters. See ExternalModel for an example of a ConfiguredModel.
    """

    @abc.abstractmethod
    def train(self, train_data: DataSet, extra_args=None):
        pass

    @abc.abstractmethod
    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        pass


class ModelConfiguration(BaseModel):
    """
    BaseClass used for configuration that a ModelTemplate takes for creating specific Models
    """

    pass
