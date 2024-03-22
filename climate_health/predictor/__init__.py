from typing import Literal

from .naive_predictor import MultiRegionPoissonModel, MultiRegionNaivePredictor

__all__ = ['MultiRegionPoissonModel', 'MultiRegionNaivePredictor']
models = __all__


def get_model(model_name: Literal[*models]):
    return globals()[model_name]
