from datetime import date, timedelta
from typing import Generic, Protocol, TypeVar

from omnipy import Dataset, Model
from omnipy.data.dataset import MultiModelDataset
from omnipy.modules.json.models import JsonDictM, JsonListM
from pydantic import BaseModel, validator


class TemporalDataModel(BaseModel):
    start_date: str | date
    period_length: int | timedelta

    @validator("start_date")
    def parse_start_date(cls, data: str | date) -> date:
        if isinstance(data, str):
            return date.fromisoformat(data)
        else:
            return data

    @validator("period_length")
    def parse_period_length_as_full_days(cls, data: int | timedelta) -> timedelta:
        if isinstance(data, int):
            return timedelta(days=data)
        else:
            assert data.seconds == 0
            assert data.microseconds == 0
            return data


class TemporalDataOmnipyDataset(
    MultiModelDataset[Model[JsonListM[TemporalDataModel]]]
): ...


class SpatioTemporalDataOmnipyDataset(
    Dataset[Model[TemporalDataOmnipyDataset]],
): ...
