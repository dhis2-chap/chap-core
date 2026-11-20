"""The principal the current request or job runs as, for recording who created what."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from sqlalchemy import event
from sqlalchemy.orm import Session

# None while authentication is off: an owner recorded then would be readable by anyone once
# authentication is turned on, so records made then have none, as in Open Geo Service.
_current_principal_id: ContextVar[str | None] = ContextVar("chap_principal_id", default=None)


def current_principal_id() -> str | None:
    return _current_principal_id.get()


def set_principal_id(principal_id: str | None) -> None:
    """Set the principal for the rest of this request; each request runs in its own context."""
    _current_principal_id.set(principal_id)


@contextmanager
def principal_scope(principal_id: str | None) -> Iterator[None]:
    """Run a block, such as a worker job, as the principal that started it."""
    token = _current_principal_id.set(principal_id)
    try:
        yield
    finally:
        _current_principal_id.reset(token)


class PrincipalLogFilter(logging.Filter):
    """Append the principal to a request log line, as ``... 200 token:caps``.

    The arguments are left alone, because uvicorn's own ``AccessFormatter`` unpacks exactly
    five of them. The principal goes into the message text, which the gunicorn configuration
    prints, and into ``record.principal`` for formatters that name it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        principal_id = _current_principal_id.get()
        record.principal = principal_id or "-"
        if principal_id is not None and isinstance(record.msg, str):
            record.msg = f"{record.msg} {principal_id.replace('%', '%%')}"
        return True


@event.listens_for(Session, "before_flush")
def _record_creator(session: Session, _flush_context, _instances) -> None:
    principal_id = _current_principal_id.get()
    if principal_id is None:
        return
    for record in session.new:
        if hasattr(record, "created_by") and record.created_by is None:
            record.created_by = principal_id
