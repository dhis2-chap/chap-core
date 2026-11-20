import pytest

from chap_core.auth.config import TierConfig
from chap_core.auth.permissions import READ, Permission
from chap_core.auth.tokens import AuthenticationConfigError, load_authentication, resolve_tiers
from chap_core.settings import ChapSettings


def tiers(**definitions: dict) -> dict[str, TierConfig]:
    return {name: TierConfig(**definition) for name, definition in definitions.items()}


def test_a_custom_tier_gets_its_includes_and_its_own_permissions():
    resolved = resolve_tiers(tiers(caps={"includes": ["read"], "permissions": ["write:datasets", "run:predictions"]}))

    assert resolved["caps"] == READ | {Permission.WRITE_DATASETS, Permission.RUN_PREDICTIONS}


def test_a_custom_tier_can_include_another_custom_tier():
    resolved = resolve_tiers(
        tiers(
            curator={"includes": ["analyst"], "permissions": ["write:models"]}, analyst={"permissions": ["read:models"]}
        )
    )

    assert resolved["curator"] == {Permission.READ_MODELS, Permission.WRITE_MODELS}


@pytest.mark.parametrize(
    ("definitions", "message"),
    [
        ({"caps": {"permissions": ["read:everything"]}}, "unknown permission read:everything"),
        ({"caps": {"includes": ["admin"]}}, "includes the unknown tier admin"),
        ({"read": {"permissions": ["read:datasets"]}}, "read is a built-in tier"),
        ({"operator": {"includes": ["read"]}}, "operator is a built-in tier"),
        ({"caps": {"permissions": ["read:*"]}}, "wildcards are not allowed"),
        ({"caps": {"permissions": ["*"]}}, "wildcards are not allowed"),
        ({"a": {"includes": ["b"]}, "b": {"includes": ["a"]}}, r"cycle \(a -> b -> a\)"),
        ({"a": {"includes": ["a"]}}, r"cycle \(a -> a\)"),
        ({"caps": {}}, "needs includes or permissions"),
    ],
)
def test_a_tier_that_cannot_work_stops_startup(definitions, message):
    with pytest.raises(AuthenticationConfigError, match=message):
        resolve_tiers(tiers(**definitions))


def test_a_token_with_a_custom_tier_gets_exactly_its_permissions(monkeypatch):
    monkeypatch.delenv("CHAP_API_TOKEN", raising=False)
    monkeypatch.setenv("CAPS_TOKEN", "c" * 32)
    block = {
        "tiers": {"caps": {"permissions": ["write:datasets"]}},
        "tokens": [{"id": "caps", "token_env": "CAPS_TOKEN", "tier": "caps"}],
    }

    (caps,) = load_authentication(ChapSettings(authentication=block)).tokens

    assert caps.permissions == {Permission.WRITE_DATASETS}


def test_a_token_naming_an_unknown_tier_stops_startup(monkeypatch):
    monkeypatch.setenv("CAPS_TOKEN", "c" * 32)
    block = {"tokens": [{"id": "caps", "token_env": "CAPS_TOKEN", "tier": "caps"}]}

    with pytest.raises(AuthenticationConfigError, match="unknown tier caps"):
        load_authentication(ChapSettings(authentication=block))
