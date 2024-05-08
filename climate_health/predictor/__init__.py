from typing import Literal, Optional

from .naive_predictor import MultiRegionPoissonModel, MultiRegionNaivePredictor
from ..external.models import models as external_models
from ..external.r_models import models as r_models

__all__ = ['MultiRegionPoissonModel', 'MultiRegionNaivePredictor']
models = __all__

ModelType = Literal[tuple(__all__ + list(external_models.keys()) + list(r_models.keys()))]
# 'MultiRegionPoissonModel', 'MultiRegionNaivePredictor', 'RegressionModel', 'HierarchicalRegressionModel']

DEFAULT_MODEL = external_models['HierarchicalModel']


def get_model(model_name: Optional[ModelType]):
    if model_name is None:
        return DEFAULT_MODEL
    if model_name in r_models:
        return r_models[model_name]
    if model_name in external_models:
        return external_models[model_name]
    return globals()[model_name]
