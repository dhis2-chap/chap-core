import os
import pytest
import yaml
import logging
from chap_core.file_io.file_paths import get_example_data_path

logger = logging.getLogger(__name__)

from chap_core.external.model_configuration import ModelTemplateConfigV2, ModelTemplateConfig
from chap_core.model_spec import (
    ModelSpec,
    model_spec_from_yaml,
    PeriodType,
    EmptyParameterSpec,
    model_spec_from_model,
)
import chap_core.predictor.feature_spec as fs
from chap_core.models.model_template import ExternalModelTemplate
from chap_core.predictor.naive_estimator import NaiveEstimator


def mlflow_paths():
    # NOTE: had to hardcode data folder here since cant use fixture as input for parametrize
    template_folder = get_example_data_path() / "model_templates"
    names = [name for name in os.listdir(template_folder) if name.endswith(".yaml")]
    return [os.path.join(template_folder, name) for name in names]


@pytest.mark.parametrize("mlflow_path", mlflow_paths())
def test_parse_configured_model_yaml_file(mlflow_path):
    data = yaml.safe_load(open(mlflow_path).read())
    m = ModelTemplateConfigV2.model_validate(data)
    logger.info(m)
    assert isinstance(m, ModelTemplateConfigV2)


