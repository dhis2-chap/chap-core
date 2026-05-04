"""
Database tables for XAI explanations.
"""

import datetime
from typing import Any

from sqlalchemy import JSON, Column
from sqlmodel import Field, Relationship

from chap_core.database.base_tables import DBModel
from chap_core.database.tables import Prediction


class PredictionExplanationBase(DBModel):
    prediction_id: int = Field(foreign_key="prediction.id", ondelete="CASCADE")
    org_unit: str
    period: str
    method: str
    output_statistic: str = "median"
    params: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    result: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    status: str = "completed"
    error: str | None = None
    created: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.UTC))


class PredictionExplanation(PredictionExplanationBase, table=True):
    id: int | None = Field(primary_key=True, default=None)
    prediction: Prediction = Relationship()
