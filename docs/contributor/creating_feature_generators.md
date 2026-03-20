# Creating Feature Generators

This guide explains how to create custom feature generators that compute derived features for Chap datasets.

## Overview

Feature generators produce computed features (e.g., cluster assignments, derived covariates) that are added to a dataset before it is passed to a model. The system provides:

- **Automatic registration**: Generators are discovered and available throughout Chap via a decorator
- **DataSet-level API**: Generators receive a full `DataSet` and return an augmented `DataSet`
- **Config-driven activation**: Models opt in to generated features using a `gen:` prefix in `required_covariates`

## Quick Start

Here is a minimal feature generator that adds a constant column:

```python
from chap_core.feature_generators import (
    FeatureGenerator,
    FeatureGeneratorSpec,
    feature_generator,
)
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


@feature_generator()
class MyConstantGenerator(FeatureGenerator):
    spec = FeatureGeneratorSpec(
        generator_id="my_constant",
        name="My Constant Feature",
        description="Adds a constant column to the dataset.",
    )

    def generate(self, dataset: DataSet) -> DataSet:
        df = dataset.to_pandas()
        df["my_constant"] = 1.0
        return DataSet.from_pandas(df)
```

A model would use this by including `gen:my_constant` in its `required_covariates`.

## FeatureGeneratorSpec

```python
from chap_core.feature_generators import FeatureGeneratorSpec

spec = FeatureGeneratorSpec(
    generator_id="unique_id",        # Used in config as gen:<generator_id>
    name="Display Name",             # Human-readable name
    description="What this generator computes",
)
```

## The generate() Method

The `generate()` method receives the full `DataSet` and must return a new `DataSet` with the generated feature(s) added as column(s). The typical pattern is:

1. Convert to pandas with `dataset.to_pandas()`
2. Compute and add columns
3. Return `DataSet.from_pandas(df)`

The generator has access to all locations and time periods, which allows cross-location features like clustering.

## Complete Example: Location Population Rank

This example ranks locations by their mean population:

```python
from chap_core.feature_generators import (
    FeatureGenerator,
    FeatureGeneratorSpec,
    feature_generator,
)
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


@feature_generator()
class PopulationRankGenerator(FeatureGenerator):
    """Ranks locations by average population."""

    spec = FeatureGeneratorSpec(
        generator_id="population_rank",
        name="Population Rank",
        description="Assigns a rank to each location based on mean population.",
    )

    def generate(self, dataset: DataSet) -> DataSet:
        df = dataset.to_pandas()
        mean_pop = df.groupby("location")["population"].mean().rank()
        df["population_rank"] = df["location"].map(mean_pop)
        return DataSet.from_pandas(df)
```

## Built-in Generator: Seasonality Cluster

The `seasonality_cluster` generator clusters locations by their normalized seasonal disease profiles using KMeans. It adds a `cluster_id` column to the dataset.

```python
from chap_core.feature_generators import get_feature_generator

cls = get_feature_generator("seasonality_cluster")
assert cls is not None
print(f"ID: {cls.spec.generator_id}")
print(f"Name: {cls.spec.name}")
```

## Registration and Discovery

### The @feature_generator() Decorator

The decorator registers your generator class when the module is imported:

```python
from chap_core.feature_generators import feature_generator, FeatureGenerator, FeatureGeneratorSpec


@feature_generator()
class RegisteredGenerator(FeatureGenerator):
    spec = FeatureGeneratorSpec(
        generator_id="registered_example",
        name="Registered Example",
        description="Example of a registered generator",
    )

    def generate(self, dataset):
        return dataset
```

### File Location

Place your generator file in `chap_core/feature_generators/` and add an import to `_discover_feature_generators()` in `chap_core/feature_generators/__init__.py`:

```console
def _discover_feature_generators():
    """Import all feature generator modules to trigger registration."""
    from chap_core.feature_generators import (
        seasonality_cluster,
        my_new_generator,  # Add your module here
    )
```

## Integration with Models

When a model lists `gen:seasonality_cluster` in its `required_covariates`, Chap automatically:

1. Parses the `gen:` prefix to identify the generator
2. Runs the generator on the dataset before passing data to the model
3. For predictions, copies location-constant generated features from historic data to future data
4. Skips `gen:` covariates during input validation (they are not expected in the raw data)

## Using the Registry

```python
from chap_core.feature_generators import (
    get_feature_generator,
    get_feature_generators_registry,
    list_feature_generators,
)

# Get a specific generator by ID
cls = get_feature_generator("seasonality_cluster")
assert cls is not None

# List all generators with metadata
for info in list_feature_generators():
    print(f"  {info['id']}: {info['name']}")

# Get full registry
registry = get_feature_generators_registry()
print(f"Registered generators: {list(registry.keys())}")
```

## Testing Your Generator

Write tests in `tests/test_feature_generators.py` using the existing `health_population_data` fixture:

```console
def test_my_generator(health_population_data):
    from chap_core.feature_generators.my_generator import MyGenerator

    generator = MyGenerator()
    result = generator.generate(health_population_data)
    assert "my_feature" in result.field_names()
```

## Reference

### Existing Implementations

| File | Description |
|------|-------------|
| `seasonality_cluster.py` | Clusters locations by seasonal disease profiles using KMeans |

### API Summary

```python
from chap_core.feature_generators import (
    feature_generator,              # Decorator to register generators
    FeatureGenerator,               # Base class (abstract)
    FeatureGeneratorSpec,           # Metadata dataclass
    get_feature_generator,          # Get generator class by ID
    get_feature_generators_registry,  # Get all registered generators
    list_feature_generators,        # List generators with metadata
    parse_generated_covariates,     # Split gen: prefixed from regular covariates
    apply_feature_generators,       # Apply generators to a dataset
)
```
