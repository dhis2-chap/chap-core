from pydantic import BaseModel
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
import abc


class ConfiguredModel(abc.ABC):
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

