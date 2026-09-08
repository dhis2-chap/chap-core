import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chap_core.database.model_templates_and_config_tables import ModelTemplateInformation
    from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


class MetaLearner(abc.ABC):
    @abc.abstractmethod
    def meta_learn(self, dataset: DataSet):
        pass

    @property
    @abc.abstractmethod
    def model_information(self) -> ModelTemplateInformation | None: ...