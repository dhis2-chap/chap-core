from typing import Literal, Optional, TYPE_CHECKING

from ..external.r_models import models as r_models

__all__ = ["MultiRegionPoissonModel", "MultiRegionNaivePredictor"]
models = __all__

all_model_names = tuple(__all__ + list(r_models.keys()))
ModelType = Literal[all_model_names]

DEFAULT_MODEL = None


def __getattr__(name: str):
    """Lazy import for heavy dependencies."""
    if name in ("MultiRegionPoissonModel", "MultiRegionNaivePredictor"):
        from .naive_predictor import MultiRegionPoissonModel, MultiRegionNaivePredictor

        globals()["MultiRegionPoissonModel"] = MultiRegionPoissonModel
        globals()["MultiRegionNaivePredictor"] = MultiRegionNaivePredictor
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_model(model_name: Optional[ModelType]):
    if model_name is None:
        return DEFAULT_MODEL
    if model_name in r_models:
        return r_models[model_name]
    # Trigger lazy import
    if model_name == "MultiRegionPoissonModel":
        from .naive_predictor import MultiRegionPoissonModel

        return MultiRegionPoissonModel
    if model_name == "MultiRegionNaivePredictor":
        from .naive_predictor import MultiRegionNaivePredictor

        return MultiRegionNaivePredictor
    raise ValueError(f"Unknown model: {model_name}")
