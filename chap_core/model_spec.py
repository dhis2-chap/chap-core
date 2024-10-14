import dataclasses
import inspect
from enum import Enum

import yaml
from pydantic import BaseModel, PositiveInt

import chap_core.predictor.feature_spec as fs
from chap_core.datatypes import TimeSeriesData

_non_feature_names = {
    "disease_cases",
    "week",
    "month",
    "location",
    "time_period",
    "year",
}


class PeriodType(Enum):
    week = "week"
    month = "month"
    any = "any"
    year = "year"


class ParameterSpec(BaseModel):
    pass


class EwarsParamSpec(ParameterSpec):
    n_weeks: PositiveInt
    alpha: float


EmptyParameterSpec = {}


class ModelSpec(BaseModel):
    name: str
    parameters: dict
    features: list[fs.Feature]
    period: PeriodType = PeriodType.any
    description: str = "No Description yet"
    author: str = "Unknown Author"
    targets: str = "disease_cases"


def model_spec_from_yaml(filename: str) -> ModelSpec:
    with open(filename, "r") as file:
        data = yaml.safe_load(file)
    name = data["name"]
    parameters = EmptyParameterSpec
    adapters = data.get("adapters", dict())
    features = [fs.feature_dict[feature] for feature in adapters.values() if feature not in _non_feature_names]
    period = PeriodType[data.get("period", "any")]
    description = data.get("description", "No Description yet")
    author = data.get("author", "Unknown Author")
    return ModelSpec(
        name=name,
        parameters=parameters,
        features=features,
        period=period,
        description=description,
        author=author,
    )


def model_spec_from_model(model_class: type) -> ModelSpec:
    name = model_class.__name__
    feature_names = _get_feature_names(model_class)
    return ModelSpec(
        name=name,
        parameters=EmptyParameterSpec,
        features=[fs.feature_dict[feature] for feature in feature_names],
        period=PeriodType.any,
        description="Internally defined model",
        author="CHAP Team",
    )


def _get_feature_names(model_class):
    var = get_dataclass(model_class)
    if var is None:
        return []
    # var = param_type.__args__[0]

    feature_names = [field.name for field in dataclasses.fields(var) if field.name not in _non_feature_names]
    return feature_names


def get_dataclass(model_class) -> type[TimeSeriesData]:
    param_type = list(inspect.get_annotations(model_class.train).values())[0]
    if not hasattr(param_type, "__args__"):
        return None
    return param_type.__args__[0]
    # return param_type
