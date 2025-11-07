from chap_core.models.configured_model import ConfiguredModel
import abc
from typing import Optional, Any
from chap_core.file_io.example_data_set import DataSetType


class HpoModelInterface(ConfiguredModel):
    @abc.abstractmethod
    def get_leaderboard(self, dataset: Optional[DataSetType]) -> list[dict[str, Any]]:
        pass

    @abc.abstractmethod
    def get_best_config(self) -> dict:
        pass
