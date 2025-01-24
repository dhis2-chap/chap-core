from typing import Optional, List

from sqlmodel import SQLModel, Field, Relationship, Column, JSON

PeriodID = str
DBModel = SQLModel
class ForecastBase(DBModel):
    period: PeriodID
    org_unit: str
    values: List[float] = Field(default_factory=list, sa_column=Column(JSON))

class SubClass(ForecastBase, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)

class OtherThing(ForecastBase, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
