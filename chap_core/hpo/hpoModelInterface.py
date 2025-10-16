from chap_core.models.configured_model import ConfiguredModel
import abc


class HpoModelInterface(ConfiguredModel):
    @abc.abstractmethod
    def get_best_config(self) -> dict:
        pass
