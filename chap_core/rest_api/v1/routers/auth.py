from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from chap_core.auth.principal import Principal

router = APIRouter(prefix="/auth", tags=["System"])


class Me(BaseModel):
    id: str = Field(
        description="The principal: `token:<id>` for a token, `open` while authentication is off, "
        "`anonymous` without a token.",
        examples=["token:dhis2-modeling-app"],
    )
    tier: str = Field(description="The principal's tier.", examples=["read"])
    permissions: list[str] = Field(description="What the caller may do, sorted.", examples=[["read:datasets"]])


@router.get("/me", response_model=Me, summary="Report who the caller is and what it may do")
def me(request: Request) -> Me:
    """Any caller may ask, so a client can hide what its token cannot do."""
    principal = request.user
    assert isinstance(principal, Principal)
    return Me(id=principal.id, tier=principal.tier, permissions=sorted(principal.permissions))
