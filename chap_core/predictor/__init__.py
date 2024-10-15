from typing import Literal, Optional

from .naive_predictor import MultiRegionPoissonModel, MultiRegionNaivePredictor

# from ..external.models import models as external_models
from ..external.r_models import models as r_models

__all__ = ["MultiRegionPoissonModel", "MultiRegionNaivePredictor"]
models = __all__

all_model_names = tuple(__all__ + list(r_models.keys()))
all_models = [d[name] for d in [globals()] for name in d.keys() if name in all_model_names]
ModelType = Literal[all_model_names]

DEFAULT_MODEL = None  # external_models['HierarchicalModel']


def get_model(model_name: Optional[ModelType]):
    if model_name is None:
        return DEFAULT_MODEL
    if model_name in r_models:
        return r_models[model_name]
    return globals()[model_name]
