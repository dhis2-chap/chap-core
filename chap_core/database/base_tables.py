from pydantic import ConfigDict
from pydantic.alias_generators import to_camel
from sqlmodel import SQLModel

PeriodID = str


class DBModel(SQLModel):
    ''' Simple wrapper that uses camelCase for the field names for the rest-api'''
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True)

    @classmethod
    def get_read_class(cls):
        ''' Returns the read class for this model'''
        class NewClass(cls):
            id: int

        NewClass.__name__ = f'{cls.__name__}Read'
        NewClass.__qualname__ = f'{cls.__qualname__}Read'
        return NewClass

