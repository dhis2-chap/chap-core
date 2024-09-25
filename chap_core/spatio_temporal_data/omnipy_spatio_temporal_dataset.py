from datetime import date
from typing import Any, Generic, TypeVar

from ..omnipy_lib import Dataset, Model, MultiModelDataset

# from omnipy.data.dataset import MultiModelDataset
from pydantic import BaseModel, root_validator, validator
from pydantic.generics import GenericModel


class TemporalDataPydanticModel(BaseModel):
    start_date: str | date = date.fromisoformat("1970-01-01")

    class Config:
        extra = "allow"

    @validator("start_date")
    def parse_start_date(cls, data: str | date) -> date:
        if isinstance(data, str):
            return date.fromisoformat(data)
        else:
            return data


TemporalDataPydanticModelT = TypeVar(
    "TemporalDataPydanticModelT", bound=TemporalDataPydanticModel
)


class TemporalDataOmnipyModel(
    Model[TemporalDataPydanticModelT], Generic[TemporalDataPydanticModelT]
):
    def to_data(self) -> Any:
        data = super().to_data()
        data["start_date"] = data["start_date"].isoformat()
        return data


class MultiResolutionTemporalDataPydanticModel(
    GenericModel, Generic[TemporalDataPydanticModelT]
):
    days: Model[list[TemporalDataOmnipyModel[TemporalDataPydanticModelT]]] = []
    weeks: Model[list[TemporalDataOmnipyModel[TemporalDataPydanticModelT]]] = []
    months: Model[list[TemporalDataOmnipyModel[TemporalDataPydanticModelT]]] = []
    inconsistent: Model[list[TemporalDataOmnipyModel[TemporalDataPydanticModelT]]] = []


class MultiResolutionTemporalDataOmnipyModel(
    Model[MultiResolutionTemporalDataPydanticModel[TemporalDataPydanticModelT]],
    Generic[TemporalDataPydanticModelT],
): ...


class TemporalSubDatasetsPydanticModel(BaseModel):
    @root_validator
    def check_all_field_types_are_temporal_omnipy_models(cls, values):
        for field in cls.__fields__.values():
            assert issubclass(field.type_, MultiResolutionTemporalDataOmnipyModel)
        return values


TemporalSubDatasetsPydanticModelT = TypeVar(
    "TemporalSubDatasetsPydanticModelT", bound=TemporalSubDatasetsPydanticModel
)


class TemporalDataOmnipyDataset(
    MultiModelDataset[
        MultiResolutionTemporalDataOmnipyModel[TemporalDataPydanticModel]
    ],
): ...


TemporalDataOmnipyDatasetT = TypeVar(
    "TemporalDataOmnipyDatasetT", bound=TemporalDataOmnipyDataset
)


class SpatioTemporalDataOmnipyDataset(
    Dataset[Model[TemporalDataOmnipyDatasetT]],
    Generic[TemporalDataOmnipyDatasetT],
): ...
