import pytest

from chap_core.model_spec import (
    model_spec_from_yaml,
    PeriodType,
    EmptyParameterSpec,
    model_spec_from_model,
)
import chap_core.predictor.feature_spec as fs
from chap_core.predictor.naive_estimator import NaiveEstimator


def test_model_spec_from_yaml(models_path):
    model_spec = model_spec_from_yaml(models_path / "ewars_Plus/config.yml")
    assert model_spec.name == "ewars_Plus"
    assert model_spec.parameters == EmptyParameterSpec
    assert model_spec.features == [fs.population, fs.rainfall, fs.mean_temperature]
    assert model_spec.period == PeriodType.week



#@pytest.mark.skip('Need a model to test')
def test_model_spec_from_yaml():
    cls = NaiveEstimator
    model_spec = model_spec_from_model(cls)
    assert model_spec.name == "NaiveEstimator"
    assert model_spec.parameters == EmptyParameterSpec
    assert set(model_spec.features) == set([]) #{fs.population, fs.rainfall, fs.mean_temperature}
