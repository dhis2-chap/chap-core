"""Base class for database data-access managers.

A manager wraps a SQLModel ``Session`` and the table model it operates on, so
entity-specific data-access code (e.g. :class:`DataSetManager`) can share common
lookups and be typed against its model. Apply the same pattern for other tables.
"""

from sqlmodel import Session, SQLModel, select


class DbManager[ModelT: SQLModel]:
    """Data-access manager bound to a single table ``model`` and a ``Session``."""

    model: type[ModelT]

    def __init__(self, session: Session):
        self.session = session

    def get(self, item_id: int) -> ModelT | None:
        """Return the row with this primary key, or ``None`` if it does not exist."""
        return self.session.get(self.model, item_id)

    def list_all(self) -> list[ModelT]:
        """Return every row for this manager's model."""
        return list(self.session.exec(select(self.model)).all())
