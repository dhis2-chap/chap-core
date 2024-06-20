from enum import Enum

import yaml
from pydantic import BaseModel

import climate_health.predictor.feature_spec as fs


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
    non_feature_names = {'disease_cases', 'week', 'month', 'location', 'time_period', 'year'}
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    name = data['name']
    parameters = EmptyParameterSpec
    adapters = data.get('adapters', dict())
    features = [fs.feature_dict[feature] for feature in adapters.values() if feature not in non_feature_names]
    period = PeriodType[data.get('period', 'any')]
    return ModelSpec(name=name, parameters=parameters, features=features, period=period)
