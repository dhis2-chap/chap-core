from climate_health.external.models.jax_models.hierarchical_model import (
    HierarchicalStateModelD2,
)
from climate_health.model_spec import (
    model_spec_from_yaml,
    PeriodType,
    EmptyParameterSpec,
    model_spec_from_model,
)
import climate_health.predictor.feature_spec as fs


def test_model_spec_from_yaml(models_path):
    model_spec = model_spec_from_yaml(models_path / "ewars_Plus/config.yml")
    assert model_spec.name == "ewars_Plus"
    assert model_spec.parameters == EmptyParameterSpec
    assert model_spec.features == [fs.population, fs.rainfall, fs.mean_temperature]
    assert model_spec.period == PeriodType.week


def test_model_spec_from_yaml():
    cls = HierarchicalStateModelD2
    model_spec = model_spec_from_model(cls)
    assert model_spec.name == "HierarchicalStateModelD2"
    assert model_spec.parameters == EmptyParameterSpec
    assert set(model_spec.features) == {fs.population, fs.rainfall, fs.mean_temperature}
