from typing import Literal

from .naive_predictor import MultiRegionPoissonModel, MultiRegionNaivePredictor
from ..external.models import models as external_models
__all__ = ['MultiRegionPoissonModel', 'MultiRegionNaivePredictor']
models = __all__

ModelType = Literal['MultiRegionPoissonModel', 'MultiRegionNaivePredictor', 'RegressionModel', 'HierarchicalRegressionModel']


def get_model(model_name: ModelType):

    if model_name in external_models:
        return external_models[model_name]
    return globals()[model_name]
