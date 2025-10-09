import inspect
from enum import Enum

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


# TODO: Move to db spec
class ModelSpec(BaseModel):
    name: str
    parameters: dict
    features: list[fs.Feature]
    period: PeriodType = PeriodType.any
    description: str = "No Description yet"
    author: str = "Unknown Author"
    targets: str = "disease_cases"


def get_dataclass(model_class) -> type[TimeSeriesData]:
    param_type = list(inspect.get_annotations(model_class.train).values())[0]
    if not hasattr(param_type, "__args__"):
        return None
    return param_type.__args__[0]
    # return param_type
