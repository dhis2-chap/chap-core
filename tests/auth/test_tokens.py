import base64

import pytest
from pydantic import ValidationError

from chap_core.auth.permissions import AUTHENTICATED, OPERATOR
from chap_core.auth.principal import ANONYMOUS
from chap_core.auth.tokens import (
    SERVICE_REGISTRY,
    AuthenticationConfigError,
    authenticate,
    load_authentication,
)
from chap_core.settings import ChapSettings

SECRET = "s" * 32
OTHER_SECRET = "o" * 32


def settings_with(*tokens: dict, **fields) -> ChapSettings:
    return ChapSettings(authentication={"tokens": list(tokens)}, **fields)


def token(id: str = "notebook", env: str = "NOTEBOOK_TOKEN", tier: str = "authenticated") -> dict:
    return {"id": id, "token_env": env, "tier": tier}


@pytest.fixture
def secrets(monkeypatch):
    monkeypatch.delenv("CHAP_API_TOKEN", raising=False)
    monkeypatch.setenv("NOTEBOOK_TOKEN", SECRET)
    monkeypatch.setenv("OTHER_TOKEN", OTHER_SECRET)


def test_a_named_token_gets_its_tier(secrets):
    authentication = load_authentication(settings_with(token()))

    (notebook,) = authentication.tokens
    assert authentication.enabled
    assert notebook.principal.id == "token:notebook"
    assert notebook.permissions == AUTHENTICATED


def test_nothing_configured_leaves_authentication_off(secrets):
    assert not load_authentication(settings_with()).enabled


def test_the_api_token_is_an_operator_token_named_legacy(secrets):
    (legacy,) = load_authentication(settings_with(api_token="short")).tokens

    assert legacy.principal.id == "token:legacy"
    assert legacy.permissions == OPERATOR


@pytest.mark.parametrize(
    ("tokens", "message"),
    [
        ([token(env="UNSET_TOKEN")], "UNSET_TOKEN is not set"),
        ([token(tier="admin")], "unknown tier admin"),
        ([token(tier="public")], "unknown tier public"),
        ([token(id="legacy")], "kept for CHAP_API_TOKEN"),
        ([token(), token(env="OTHER_TOKEN")], "the id notebook is used twice"),
        ([token(), token(id="copy")], "notebook and copy have the same secret"),
    ],
)
def test_a_configuration_that_cannot_work_stops_startup(secrets, tokens, message):
    with pytest.raises(AuthenticationConfigError, match=message):
        load_authentication(settings_with(*tokens))


def test_a_short_secret_stops_startup(secrets, monkeypatch):
    monkeypatch.setenv("NOTEBOOK_TOKEN", "short")

    with pytest.raises(AuthenticationConfigError, match="shorter than 32 characters"):
        load_authentication(settings_with(token()))


def test_the_api_token_may_not_repeat_a_named_secret(secrets):
    with pytest.raises(AuthenticationConfigError, match="same secret"):
        load_authentication(settings_with(token(), api_token=SECRET))


def test_an_open_geo_service_block_loads_unchanged(secrets):
    block = {"tokens": [{"id": "caps", "token_env": "NOTEBOOK_TOKEN", "tier": "operator"}]}

    (caps,) = load_authentication(ChapSettings(authentication=block)).tokens

    assert caps.principal.id == "token:caps"


@pytest.fixture
def authentication(secrets):
    return load_authentication(settings_with(token(), service_registration_key="r" * 32))


def basic(id: str, secret: str) -> str:
    return "Basic " + base64.b64encode(f"{id}:{secret}".encode()).decode()


@pytest.mark.parametrize(
    "header",
    [f"Bearer {SECRET}", f"bearer basic//{SECRET}", f"ApiToken {SECRET}", basic("notebook", SECRET)],
)
def test_each_accepted_scheme_names_the_token(authentication, header):
    principal = authenticate(header, None, "/v1/crud/datasets", authentication)

    assert principal is not None and principal.id == "token:notebook"


@pytest.mark.parametrize(
    "header",
    [f"Bearer {OTHER_SECRET}", basic("someone-else", SECRET), "Basic not-base64!", "Basic é", f"Token {SECRET}"],
)
def test_a_credential_that_matches_no_token_is_invalid(authentication, header):
    assert authenticate(header, None, "/v1/crud/datasets", authentication) is None


def test_a_secret_in_the_wrong_setting_is_not_echoed_in_the_startup_error():
    block = {"tokens": [{"id": "x", "token_env": "pasted/secret", "tier": "read"}]}

    with pytest.raises(ValidationError) as error:
        ChapSettings(authentication=block)

    assert "pasted/secret" not in str(error.value)


def test_no_credential_is_anonymous(authentication):
    assert authenticate(None, None, "/v1/crud/datasets", authentication) is ANONYMOUS


def test_the_registration_key_opens_only_the_service_registry(authentication):
    assert authenticate(None, "r" * 32, "/v2/services/$register", authentication) is SERVICE_REGISTRY
    assert authenticate(None, "r" * 32, "/v1/crud/datasets", authentication) is None


def test_a_valid_service_key_is_used_when_authorization_is_invalid(authentication):
    principal = authenticate("Bearer wrong", SECRET, "/v1/crud/datasets", authentication)

    assert principal is not None and principal.id == "token:notebook"
