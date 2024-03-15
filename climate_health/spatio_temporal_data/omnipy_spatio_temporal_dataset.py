from datetime import date
from typing import Generic, TypeVar

from omnipy import Dataset, Model
from omnipy.modules.json.models import JsonListM
from pydantic import BaseModel, validator


class TemporalDataModel(BaseModel):
    start_date: str | date

    @validator("start_date")
    def parse_start_date(cls, data: str | date) -> date:
        if isinstance(data, str):
            return date.fromisoformat(data)
        else:
            return data


TemporalDataModelT = TypeVar("TemporalDataModelT", bound=TemporalDataModel)


class TemporalDataOmnipyModel(
    Model[JsonListM[TemporalDataModelT]], Generic[TemporalDataModelT]
): ...


TemporalDataPydanticModelT = TypeVar("TemporalDataPydanticModelT", bound=BaseModel)


class SpatioTemporalDataOmnipyDataset(
    Dataset[Model[TemporalDataPydanticModelT]],
    Generic[TemporalDataPydanticModelT],
):
    def __new__(cls, *args, **kwargs):
        pydantic_model = cls.get_model_class().outer_type()
        for field in pydantic_model.__fields__.values():
            assert issubclass(field.type_, TemporalDataOmnipyModel)

        return super().__new__(cls)
