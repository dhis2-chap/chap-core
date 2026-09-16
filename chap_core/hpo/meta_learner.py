import abc

from chap_core.database.model_templates_and_config_tables import ModelTemplateInformation
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


class MetaLearner(abc.ABC):
    """
    A MetaLearner is the stage of a model between model template and configured model.
    """

    @abc.abstractmethod
    def meta_learn(self, dataset: DataSet):
        pass

    @property
    @abc.abstractmethod
    def model_information(self) -> ModelTemplateInformation | None: ...
