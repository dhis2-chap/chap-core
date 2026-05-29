"""Base class for database data-access managers.

A manager wraps a SQLModel ``Session`` and the table model it operates on, so
entity-specific data-access code (e.g. :class:`DataSetManager`) can share common
lookups and be typed against its model. Method names mirror the servicekit manager
interface (``save`` / ``find_by_id`` / ``find_all`` / ``delete_by_id``) so chap-core's
sync data-access reads consistently with the rest of the stack. Apply the same
pattern for other tables.
"""

from sqlmodel import Session, SQLModel, select


class DbManager[ModelT: SQLModel]:
    """Data-access manager bound to a single table ``model`` and a ``Session``."""

    model: type[ModelT]

    def __init__(self, session: Session):
        self.session = session

    def find_by_id(self, item_id: int) -> ModelT | None:
        """Return the row with this primary key, or ``None`` if it does not exist."""
        return self.session.get(self.model, item_id)

    def find_all(self) -> list[ModelT]:
        """Return every row for this manager's model."""
        return list(self.session.exec(select(self.model)).all())

    def save(self, obj: ModelT) -> ModelT:
        """Persist a row and return the committed instance."""
        self.session.add(obj)
        self.session.commit()
        return obj

    def delete_by_id(self, item_id: int) -> None:
        """Delete the row with this primary key, if it exists."""
        obj = self.find_by_id(item_id)
        if obj is not None:
            self.session.delete(obj)
            self.session.commit()
