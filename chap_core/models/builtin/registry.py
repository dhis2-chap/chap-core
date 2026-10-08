"""
Registry of built-in models.

Built-in models are Python models that run in-process inside chap-core, without
Docker, an MLproject or a chapkit service. They are registered with
``@builtin_model()`` and reached through ``BuiltinModelTemplate``, which gives
them the same interface as any other model template.

Rules for built-in models:

- ``predict`` returns samples for exactly the future periods of every location, and
  every sample is finite.
- No state is shared between instances, and no files are written.
- Any randomness uses a fixed seed.
"""

import abc
from dataclasses import dataclass
from typing import ClassVar

from chap_core.database.model_templates_and_config_tables import ModelTemplateRole
from chap_core.model_spec import PeriodType
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

BUILTIN_SOURCE_PREFIX = "builtin:"


@dataclass(frozen=True)
class BuiltinModelSpec:
    """What chap-core stores as the model template of a built-in model.

    A template version is write-once, so bump ``version`` when any of these change.
    """

    name: str
    display_name: str
    description: str
    version: str = "1"
    supported_period_type: PeriodType = PeriodType.any
    required_covariates: tuple[str, ...] = ()
    role: ModelTemplateRole | None = None


class BuiltinModel(abc.ABC):
    """Base class for built-in models. A new instance is created for every run."""

    spec: ClassVar[BuiltinModelSpec]

    @abc.abstractmethod
    def train(self, data: DataSet) -> None:
        """Fit the model on training data with a ``disease_cases`` column."""

    @abc.abstractmethod
    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        """Return a ``DataSet[Samples]`` covering the periods and locations of ``future_data``."""


_builtin_model_registry: dict[str, type[BuiltinModel]] = {}


def builtin_model():
    """Decorator to register a built-in model under ``spec.name``."""

    def decorator(cls: type[BuiltinModel]) -> type[BuiltinModel]:
        name = cls.spec.name
        if name in _builtin_model_registry:
            raise ValueError(f"Built-in model {name!r} is already registered")
        _builtin_model_registry[name] = cls
        return cls

    return decorator


def get_builtin_models() -> dict[str, type[BuiltinModel]]:
    """All registered built-in models by name."""
    return _builtin_model_registry.copy()


def get_builtin_model(name: str) -> type[BuiltinModel]:
    """Get a registered built-in model by name."""
    if name not in _builtin_model_registry:
        raise ValueError(f"Unknown built-in model {name!r}. Available: {sorted(_builtin_model_registry)}")
    return _builtin_model_registry[name]
