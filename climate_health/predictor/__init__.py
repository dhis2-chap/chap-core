from typing import Literal

from .naive_predictor import MultiRegionPoissonModel, MultiRegionNaivePredictor

__all__ = ['MultiRegionPoissonModel', 'MultiRegionNaivePredictor']
models = __all__

ModelType = Literal['MultiRegionPoissonModel', 'MultiRegionNaivePredictor']


def get_model(model_name: ModelType):
    return globals()[model_name]
