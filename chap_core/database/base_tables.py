from pydantic import ConfigDict
from pydantic.alias_generators import to_camel
from sqlmodel import SQLModel

PeriodID = str


class DBModel(SQLModel):
    ''' Simple wrapper that uses camelCase for the field names for the rest-api'''
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True)
