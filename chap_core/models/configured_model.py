from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from chap_core.database.model_templates_and_config_tables import ModelTemplateInformation
    from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


class ConfiguredModel(abc.ABC):
    """
    A ConfiguredModel is the main interface for all models in the Chap framework.
    A configured model is different from a model template in that it is configured with specific hyperparameters
    and/or other choices. While a ModelTemplate is flexible with choices, a ConfiguredModel has fixed choices
    and parameters. See ExternalModel for an example of a ConfiguredModel.
    """

    @property
    @abc.abstractmethod
    def model_information(self) -> ModelTemplateInformation | None: ...

    @abc.abstractmethod
    def train(self, train_data: DataSet, extra_args=None):
        pass

    @abc.abstractmethod
    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        pass

    def model_info(self) -> ModelInfo | None:
        """Returns the models ModelInfo, or None if the model does not specify one"""
        return None


class ModelConfiguration(BaseModel):
    """
    BaseClass used for configuration that a ModelTemplate takes for creating specific Models
    """


class ModelInfo(BaseModel, extra="ignore"):
    """
    ModelInfo holds information a configured model shares through its model_info entry point.

    All fields are optional, and unknown fields are ignored. New fields can be added without
    needing to change existing model_info implementations.

    context_length is an int or None, and represents how many timesteps backwards from the most recent timestep
    the model uses. None (null or missing value in model_info.json) represents
    all available history in the given dataset. Values that are <= 0 are not allowed.
    """

    context_length: int | None = Field(default=None, gt=0)
