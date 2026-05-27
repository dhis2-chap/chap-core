"""
Feature generators submodule.

Provides a registry-based system for generating derived features on DataSets.
Generated features are specified in model configs via `gen:` prefix in
required_covariates, e.g. `gen:seasonality_cluster`.

Generators are registered using the @feature_generator() decorator.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

logger = logging.getLogger(__name__)

GEN_PREFIX = "gen:"


@dataclass
class FeatureGeneratorSpec:
    generator_id: str
    name: str
    description: str


class FeatureGenerator(ABC):
    spec: FeatureGeneratorSpec

    @abstractmethod
    def generate(self, dataset: DataSet) -> DataSet:
        """Generate derived features and return an augmented DataSet."""


_generators_registry: dict[str, type[FeatureGenerator]] = {}


def feature_generator():
    """Decorator to register a feature generator class."""

    def decorator(cls: type[FeatureGenerator]) -> type[FeatureGenerator]:
        if not isinstance(cls, type) or not issubclass(cls, FeatureGenerator):
            raise TypeError(f"{cls} must be a class inheriting from FeatureGenerator")
        if not hasattr(cls, "spec"):
            raise ValueError(f"{cls.__name__} missing 'spec' class attribute")
        _generators_registry[cls.spec.generator_id] = cls
        return cls

    return decorator


def get_feature_generators_registry() -> dict[str, type[FeatureGenerator]]:
    """Get a copy of the feature generators registry."""
    return _generators_registry.copy()


def get_feature_generator(generator_id: str) -> type[FeatureGenerator] | None:
    """Get a feature generator class by ID."""
    return _generators_registry.get(generator_id)


def list_feature_generators() -> list[dict]:
    """List all registered feature generators with metadata."""
    return [
        {
            "id": cls.spec.generator_id,
            "name": cls.spec.name,
            "description": cls.spec.description,
        }
        for cls in _generators_registry.values()
    ]


def parse_generated_covariates(required_covariates: list[str]) -> tuple[list[str], list[str]]:
    """Split required_covariates into regular covariates and generator IDs.

    Returns
    -------
    tuple[list[str], list[str]]
        (regular_covariates, generator_ids)
    """
    regular = []
    generator_ids = []
    for cov in required_covariates:
        if cov.startswith(GEN_PREFIX):
            generator_ids.append(cov[len(GEN_PREFIX) :])
        else:
            regular.append(cov)
    return regular, generator_ids


def apply_feature_generators(dataset: DataSet, generator_ids: list[str]) -> DataSet:
    """Apply the specified feature generators to a dataset.

    Raises
    ------
    ValueError
        If a generator_id is not found in the registry.
    """
    for gid in generator_ids:
        cls = _generators_registry.get(gid)
        if cls is None:
            available = ", ".join(_generators_registry.keys())
            raise ValueError(f"Unknown feature generator '{gid}'. Available: {available}")
        generator = cls()
        dataset = generator.generate(dataset)
    return dataset


def _discover_feature_generators():
    """Import all feature generator modules to trigger registration."""
    from chap_core.feature_generators import seasonality_cluster


_discover_feature_generators()
