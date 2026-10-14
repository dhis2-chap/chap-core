from chap_core.models.builtin import global_median, persistence, seasonal_median
from chap_core.models.builtin.registry import (
    BUILTIN_SOURCE_PREFIX,
    BaselineNotApplicable,
    BuiltinModel,
    BuiltinModelSpec,
    builtin_model,
    get_builtin_model,
    get_builtin_models,
)
from chap_core.models.builtin.template import BuiltinConfiguredModel, BuiltinModelTemplate, builtin_template_config

__all__ = [
    "BUILTIN_SOURCE_PREFIX",
    "BaselineNotApplicable",
    "BuiltinConfiguredModel",
    "BuiltinModel",
    "BuiltinModelSpec",
    "BuiltinModelTemplate",
    "builtin_model",
    "builtin_template_config",
    "get_builtin_model",
    "get_builtin_models",
    "global_median",
    "persistence",
    "seasonal_median",
]
