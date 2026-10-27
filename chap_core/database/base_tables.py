import datetime
from typing import Annotated

from pydantic import ConfigDict, PlainSerializer, create_model
from pydantic.alias_generators import to_camel
from sqlmodel import SQLModel

PeriodID = str


def utc_now() -> datetime.datetime:
    """The current time in UTC, without a time zone, as the datetime columns store it."""
    return datetime.datetime.now(datetime.UTC).replace(tzinfo=None)


def _serialize_utc(value: datetime.datetime) -> str:
    if value.tzinfo is None:
        return value.replace(tzinfo=datetime.UTC).isoformat()
    return value.astimezone(datetime.UTC).isoformat()


# The columns hold naive UTC, so the API adds the offset that clients need to show local time.
UtcDatetime = Annotated[datetime.datetime, PlainSerializer(_serialize_utc, when_used="json")]


class DBModel(SQLModel):
    """Simple wrapper that uses camelCase for the field names for the rest-api"""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)  # type: ignore[assignment]

    @classmethod
    def get_read_class(cls):
        """Returns the read class for this model"""

        class NewClass(cls):  # type: ignore[valid-type, misc]
            id: int

        NewClass.__name__ = f"{cls.__name__}Read"
        NewClass.__qualname__ = f"{cls.__qualname__}Read"
        return NewClass

    @classmethod
    def get_create_class(cls):
        """Remove the id field from the class"""
        # create
        fields = {name: (field.annotation, field.default) for name, field in cls.model_fields.items() if name != "id"}

        NewModel = create_model(f"{cls.__name__}Create", **fields)  # type: ignore[call-overload]
        return NewModel
