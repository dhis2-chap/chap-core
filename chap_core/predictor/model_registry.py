from typing import Literal

from .naive_estimator import NaiveEstimator
from .published_models import model_dict
from ..external.external_model import get_model_from_directory_or_github_url
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


registry = ModelRegistry(model_dict)
