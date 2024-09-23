from typing import List

from pydantic import BaseModel


class PredictionBase(BaseModel):
    orgUnit: str
    dataElement: str
    period: str


class PredictionResponse(PredictionBase):
    value: float


class PredictionSamplResponse(PredictionBase):
    values: list[float]


class FullPredictionResponse(BaseModel):
    diseaseId: str
    dataValues: List[PredictionResponse]


class FullPredictionSampleResponse(BaseModel):
    diseaseId: str
    dataValues: List[PredictionSamplResponse]
