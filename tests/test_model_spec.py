from climate_health.model_spec import model_spec_from_yaml, PeriodType, EmptyParameterSpec
import climate_health.predictor.feature_spec as fs


def test_model_spec_from_yaml(models_path):
    model_spec = model_spec_from_yaml(models_path / 'ewars_Plus/config.yml')
    assert model_spec.name == 'ewars_Plus'
    assert model_spec.parameters == EmptyParameterSpec
    assert model_spec.features == [fs.population, fs.rainfall, fs.mean_temperature]
    assert model_spec.period == PeriodType.week
