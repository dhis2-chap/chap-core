from pydantic import ConfigDict, create_model
from pydantic.alias_generators import to_camel
from sqlmodel import SQLModel

PeriodID = str


class DBModel(SQLModel):
    """Simple wrapper that uses camelCase for the field names for the rest-api"""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    @classmethod
    def get_read_class(cls):
        """Returns the read class for this model"""

        class NewClass(cls):
            id: int

        NewClass.__name__ = f"{cls.__name__}Read"
        NewClass.__qualname__ = f"{cls.__qualname__}Read"
        return NewClass

    @classmethod
    def get_create_class(cls):
        """Remove the id field from the class"""
        # create
        fields = {name: (field.annotation, field.default) for name, field in cls.model_fields.items() if name != "id"}

        NewModel = create_model(f"{cls.__name__}Create", **fields)
        return NewModel
