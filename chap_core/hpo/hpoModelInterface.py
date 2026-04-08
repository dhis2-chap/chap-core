import abc
from typing import Any

from chap_core.models.configured_model import ConfiguredModel
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


class HpoModelInterface(ConfiguredModel):
    @abc.abstractmethod
    # def get_leaderboard(self, dataset: DataSetType | None) -> list[dict[str, Any]]:
    def get_leaderboard(self, dataset: DataSet) -> list[dict[str, Any]]:
        pass

    @abc.abstractmethod
    def get_best_config(self) -> dict:
        pass
