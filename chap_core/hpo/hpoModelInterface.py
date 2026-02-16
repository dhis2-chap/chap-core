import abc
from typing import Any

from chap_core.file_io.example_data_set import DataSetType
from chap_core.models.configured_model import ConfiguredModel


class HpoModelInterface(ConfiguredModel):
    @abc.abstractmethod
    def get_leaderboard(self, dataset: DataSetType | None) -> list[dict[str, Any]]:
        pass

    @abc.abstractmethod
    def get_best_config(self) -> dict:
        pass
