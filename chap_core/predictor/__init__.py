from typing import Literal, Optional, TypeAlias

# from ..external.models import models as external_models
from ..external.r_models import models as r_models
from .naive_predictor import MultiRegionNaivePredictor, MultiRegionPoissonModel

__all__ = ["MultiRegionNaivePredictor", "MultiRegionPoissonModel"]
models = __all__

all_model_names = tuple(__all__ + list(r_models.keys()))
all_models = [d[name] for d in [globals()] for name in d if name in all_model_names]
type ModelType = Literal["MultiRegionPoissonModel", "MultiRegionNaivePredictor", "ewars_Plus"]

DEFAULT_MODEL = None  # external_models['HierarchicalModel']


def get_model(model_name: ModelType | None):
    if model_name is None:
        return DEFAULT_MODEL
    if model_name in r_models:
        return r_models[model_name]
    return globals()[model_name]
