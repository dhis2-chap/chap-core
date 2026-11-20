"""The ``authentication`` block of ``chap-server.yaml``, in Open Geo Service's format."""

from pydantic import BaseModel, ConfigDict, Field

# Shared by token ids and tier names.
NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]*$"


class TokenConfig(BaseModel):
    # A secret pasted into token_env by mistake must not be echoed in the startup error.
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str = Field(pattern=NAME_PATTERN)
    token_env: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    tier: str


class TierConfig(BaseModel):
    """A custom tier: the permissions of the tiers it includes, plus its own."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    includes: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ()


class AuthenticationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    # Chap's addition to Open Geo Service's format.
    tiers: dict[str, TierConfig] = Field(default_factory=dict)
    tokens: tuple[TokenConfig, ...] = ()
