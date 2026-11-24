import pytest

from chap_core.auth.context import current_principal_id, principal_scope, without_principal


def test_shared_work_runs_as_no_principal():
    @without_principal
    def who() -> str | None:
        return current_principal_id()

    with principal_scope("token:caps"):
        assert who() is None
        assert current_principal_id() == "token:caps"


def test_a_refused_request_does_not_keep_the_previous_principal(monkeypatch):
    """Requests pipelined on one connection share a context, so the backend clears it first."""
    import asyncio

    from starlette.authentication import AuthenticationError
    from starlette.requests import HTTPConnection

    from chap_core.auth.context import set_principal_id
    from chap_core.rest_api.auth import ChapTokenBackend

    monkeypatch.setenv("CHAP_API_TOKEN", "a" * 32)
    monkeypatch.delenv("CHAP_CONFIG", raising=False)
    refused = HTTPConnection({"type": "http", "method": "GET", "path": "/v1/jobs", "headers": []})

    async def previous_then_refused() -> str | None:
        set_principal_id("token:legacy")
        with pytest.raises(AuthenticationError):
            await ChapTokenBackend().authenticate(refused)
        return current_principal_id()

    assert asyncio.run(previous_then_refused()) is None


def test_a_principal_may_act_only_on_what_it_created():
    from chap_core.auth.permissions import ANALYST, ADMIN
    from chap_core.auth.principal import Principal

    caps = Principal(id="token:caps", tier="analyst", permissions=ANALYST)
    admin = Principal(id="token:ops", tier="admin", permissions=ADMIN)

    assert caps.may_act_on("token:caps")
    assert not caps.may_act_on("token:other")
    assert not caps.may_act_on(None)
    assert admin.may_act_on("token:other") and admin.may_act_on(None)
