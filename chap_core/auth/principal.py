"""Who a request runs as, and what it may do."""

from dataclasses import dataclass

from starlette.authentication import BaseUser

from chap_core.auth.permissions import OPERATOR, PUBLIC, Permission


@dataclass(frozen=True)
class Principal(BaseUser):
    """A caller: a token, a legacy key, or nobody. Every principal has exactly one tier."""

    id: str
    tier: str
    permissions: frozenset[Permission]

    @property
    def is_authenticated(self) -> bool:
        return self is not ANONYMOUS

    @property
    def display_name(self) -> str:
        return self.id

    @property
    def identity(self) -> str:
        return self.id


ANONYMOUS = Principal(id="anonymous", tier="public", permissions=PUBLIC)
"""A caller without a credential while authentication is on."""

OPEN = Principal(id="open", tier="operator", permissions=OPERATOR)
"""Every caller while authentication is off."""
