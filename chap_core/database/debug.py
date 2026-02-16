from typing import Optional

from sqlmodel import Field

from chap_core.database.base_tables import DBModel


class DebugEntry(DBModel, table=True):
    id: int | None = Field(primary_key=True, default=None)
    timestamp: float
