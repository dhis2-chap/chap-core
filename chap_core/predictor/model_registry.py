from pathlib import Path
from typing import Literal
import yaml

from .naive_estimator import NaiveEstimator
from .published_models import model_dict
from ..models.utils import get_model_from_directory_or_github_url
from ..model_spec import PeriodType, ModelSpec
import logging

logger = logging.getLogger(__name__)

naive_spec = ModelSpec(
    name="naive_model",
    parameters={},
    features=[],
    period=PeriodType.any,
    description="Naive model used for testing",
    author="CHAP",
)


class ModelRegistry:
    def __init__(self, model_dict: dict):
        self._model_type = Literal[("naive_model",) + tuple(model_dict.keys())]
        self._model_specs = [naive_spec, *model_dict.values()]

    @property
    def model_type(self):
        return self._model_type

    def get_model(self, model_id: str, ignore_env=False):
        logger.info(f"Getting model with id: {model_id}")
        if model_id == "naive_model":
            return NaiveEstimator()
        elif model_id in model_dict:
            spec = model_dict[model_id]
            return get_model_from_directory_or_github_url(spec.github_link, ignore_env=ignore_env)
        else:
            raise ValueError(f"Unknown model id: {model_id}, expected one of 'naive_model', {list(model_dict.keys())}")

    def list_specifications(self):
        return self._model_specs

    @classmethod
    def from_model_templates_config_file(cls, yaml_config_file: Path):
        """Reads a list of models by reading model teamplates from a local config file.
        Each line in the file should have key: values corresponding
        to model name (id) and github url.
        This method converts each model template to a model by assuming default choices to the template.
        """
        with open(yaml_config_file, "r") as file:
            data = yaml.safe_load(file)
        model_dict = {}
        for model_name, github_url in data.items():
            # url = giturlparse.parse(github_url)
            # owner = url.owner
            # name = url.name
            model_dict[model_name] = ModelSpec(
                name=model_name,
                parameters={},
                features=[],
                period=PeriodType.any,
                description="Model from config file",
                author="CHAP",
                github_link=github_url,
            )
        return cls(model_dict)


registry = ModelRegistry(model_dict)
