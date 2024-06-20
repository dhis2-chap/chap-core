import dataclasses
import inspect
from enum import Enum

import yaml
from pydantic import BaseModel

import climate_health.predictor.feature_spec as fs

_non_feature_names = {'disease_cases', 'week', 'month', 'location', 'time_period', 'year'}

class PeriodType(Enum):
    week = "week"
    month = "month"
    any = "any"
    year = "year"


class ParameterSpec(BaseModel):
    pass


class EmptyParameterSpec(ParameterSpec):
    ...


class ModelSpec(BaseModel):
    name: str
    parameters: type[ParameterSpec]
    features: list[fs.Feature]
    period: PeriodType = PeriodType.any


def model_spec_from_yaml(filename: str) -> ModelSpec:

    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    name = data['name']
    parameters = EmptyParameterSpec
    adapters = data.get('adapters', dict())
    features = [fs.feature_dict[feature] for feature in adapters.values() if feature not in _non_feature_names]
    period = PeriodType[data.get('period', 'any')]
    return ModelSpec(name=name, parameters=parameters, features=features, period=period)

def model_spec_from_model(model_class: type) -> ModelSpec:
    name = model_class.__name__
    parameters = EmptyParameterSpec
    param_type = list(inspect.get_annotations(model_class.train).values())[0]

    # get the generic typevar and get the type of the first argument
    var = param_type.__args__[0]
    print(var)

    feature_names = [field.name for field in dataclasses.fields(var) if field.name not in _non_feature_names]
    print(feature_names)
    return ModelSpec(name=name,
                     parameters=EmptyParameterSpec,
                     features=[fs.feature_dict[feature] for feature in feature_names],
                     period=PeriodType.any
                     )
