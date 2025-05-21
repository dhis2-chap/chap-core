import numpy as np
import pytest
from pydantic import BaseModel

from chap_core import ModelTemplateInterface
from chap_core.adaptors.command_line_interface import generate_template_app
from chap_core.database.model_templates_and_config_tables import ModelTemplateInformation, ModelConfiguration
from chap_core.datatypes import Samples
from chap_core.external.model_configuration import ModelTemplateConfigCommon
from chap_core.models.configured_model import ConfiguredModel
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
import logging

logger = logging.getLogger(__name__)


class DummyConfig(BaseModel):
    n_iterations: int = 10


class DummyModel(ConfiguredModel):
    covariate_names: list[str] = ['rainfall', 'mean_temperature', 'population']

    def __init__(self, config: dict):
        self._config = DummyConfig.model_validate(config.user_option_values)

    def save(self, filepath):
        with open(filepath, 'w') as f:
            f.write(self._config.model_dump_json())

    @classmethod
    def load_predictor(cls, filepath):
        return cls(ModelConfiguration.parse_file(filepath))

    def train(self, train_data: DataSet):
        logger.info(f'Training with {self._config}')
        return self

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        logger.info(f'Prediction with {self._config}')
        return DataSet({location: Samples(future_data.period_range, np.full((len(future_data.period_range), 100), 10.))
                        for location in future_data.keys()})


class DummyModelTemplate(ModelTemplateInterface):
    model_template_info = ModelTemplateConfigCommon()

    def get_config_class(self) -> type[ModelConfiguration]:
        return DummyConfig

    def get_schema(self) -> ModelTemplateInformation:
        return DummyConfig.model_json_schema()['properties']

    def get_model(self, model_configuration: ModelConfiguration = None) -> ConfiguredModel:
        return DummyModel(model_configuration)


@pytest.fixture()
def model_config():
    return DummyConfig(n_iterations=10)


@pytest.fixture
def model_config_path(tmp_path, model_config):
    path = tmp_path / 'model_config.json'
    with open(path, 'w') as f:
        f.write(model_config.model_dump_json())
    return path


def test_generate_template_app(dumped_weekly_data_paths, tmp_path, model_config_path):
    app, train, predict, _ = generate_template_app(DummyModelTemplate())
    training_path, historic_path, future_path = dumped_weekly_data_paths
    model_path = tmp_path / 'model'
    train(training_path, model_path, model_config_path)
    predict(model_path, historic_path, future_path, tmp_path/'predictions.csv', model_config_path)

def test_generate_template_app_yaml(model_config_path):
    *_, write_yaml = generate_template_app(DummyModelTemplate())
    write_yaml()
