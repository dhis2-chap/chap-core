"""Named tokens from the settings, and the principal a credential names."""

import base64
import hmac
import logging
import os
import re
from dataclasses import dataclass, field

from chap_core.auth.config import NAME_PATTERN, TierConfig
from chap_core.auth.permissions import ADMIN, BUILTIN_TIERS, Permission
from chap_core.auth.principal import ANONYMOUS, Principal
from chap_core.settings import ChapSettings, get_settings

logger = logging.getLogger(__name__)

MIN_SECRET_LENGTH = 32
LEGACY_TOKEN_ID = "legacy"  # noqa: S105 — a token id, not a secret

# servicekit can only send X-Service-Key, so the registration key is accepted there, on the
# service registry only, so it cannot be used as a general API credential.
SERVICE_REGISTRY_PREFIX = "/v2/services"

SERVICE_REGISTRY = Principal(
    id="service-registry",
    tier="service-registry",
    permissions=frozenset({Permission.REGISTER_SERVICES, Permission.READ_SERVICES, Permission.CALL_SERVICES}),
)
"""A chapkit service holding ``SERVICEKIT_REGISTRATION_KEY``: the registry routes it opens today."""


class AuthenticationConfigError(ValueError):
    """The authentication settings would leave a client unable to sign in, or two clients alike."""


@dataclass(frozen=True)
class NamedToken:
    id: str
    tier: str
    permissions: frozenset[Permission]
    secret: str = field(repr=False)

    @property
    def principal(self) -> Principal:
        return Principal(id=f"token:{self.id}", tier=self.tier, permissions=self.permissions)


@dataclass(frozen=True)
class Authentication:
    """The resolved tokens, and the registration key, for the current settings."""

    tokens: tuple[NamedToken, ...]
    registration_key: str | None = field(default=None, repr=False)

    @property
    def enabled(self) -> bool:
        return bool(self.tokens)


def _variable(name: str) -> str:
    """A token_env for an error message, unless it may be a secret pasted there by mistake."""
    return name if re.fullmatch(r"[A-Z_][A-Z0-9_]*", name) else "the variable token_env names"


def _tier(name: str) -> str:
    """A tier name for an error message, unless it is as long as a secret and may be one."""
    return name if len(name) < MIN_SECRET_LENGTH else "(a name as long as a secret, not shown)"


def load_authentication(settings: ChapSettings) -> Authentication:
    """Resolve the tiers and the token secrets, refusing a configuration that cannot work."""
    tiers = resolve_tiers(settings.authentication.tiers)
    tokens: list[NamedToken] = []
    for index, item in enumerate(settings.authentication.tokens):
        where = f"authentication.tokens[{index}] ({item.id})"
        secret = os.environ.get(item.token_env, "").strip()
        if not secret:
            raise AuthenticationConfigError(f"{where}: environment variable {_variable(item.token_env)} is not set")
        if len(secret) < MIN_SECRET_LENGTH:
            raise AuthenticationConfigError(
                f"{where}: the secret in {_variable(item.token_env)} is shorter than {MIN_SECRET_LENGTH} characters"
            )
        if item.id == LEGACY_TOKEN_ID:
            raise AuthenticationConfigError(f"{where}: the id {LEGACY_TOKEN_ID} is kept for CHAP_API_TOKEN")
        if item.tier not in tiers or item.tier == "public":
            raise AuthenticationConfigError(f"{where}: unknown tier {_tier(item.tier)}")
        tokens.append(NamedToken(id=item.id, tier=item.tier, permissions=tiers[item.tier], secret=secret))

    if settings.api_token is not None:
        legacy = NamedToken(
            id=LEGACY_TOKEN_ID, tier="admin", permissions=ADMIN, secret=settings.api_token.get_secret_value()
        )
        tokens.append(legacy)

    _refuse_repeats(tokens)
    key = settings.service_registration_key.get_secret_value() if settings.service_registration_key else None
    named = [token for token in tokens if token.id != LEGACY_TOKEN_ID]
    if key is not None and any(secret_matches(key, token.secret) for token in named):
        # X-Service-Key would name the token, not the registry, so one secret must not be both.
        # CHAP_API_TOKEN is exempt: installs set both to one secret, as the docs once said for
        # chapkit, and that secret names token:legacy, which holds every registry permission.
        raise AuthenticationConfigError(
            "SERVICEKIT_REGISTRATION_KEY is also the secret of a named token; use separate secrets"
        )
    return Authentication(tokens=tuple(tokens), registration_key=key)


def resolve_tiers(custom: dict[str, TierConfig]) -> dict[str, frozenset[Permission]]:
    """The built-in tiers and the custom ones, each as a flat set of permissions."""
    known = {permission.value for permission in Permission}
    for name, tier in custom.items():
        where = f"authentication.tiers.{name}"
        if name in BUILTIN_TIERS:
            raise AuthenticationConfigError(f"{where}: {name} is a built-in tier and cannot be redefined")
        if not re.match(NAME_PATTERN, name):
            raise AuthenticationConfigError(f"{where}: a tier name takes letters, digits, - and _")
        if not tier.includes and not tier.permissions:
            raise AuthenticationConfigError(f"{where}: needs includes or permissions")
        for permission in tier.permissions:
            if "*" in permission:
                raise AuthenticationConfigError(f"{where}: wildcards are not allowed ({permission})")
            if permission not in known:
                raise AuthenticationConfigError(f"{where}: unknown permission {permission}")
        for included in tier.includes:
            if included not in BUILTIN_TIERS and included not in custom:
                raise AuthenticationConfigError(f"{where}: includes the unknown tier {included}")

    resolved: dict[str, frozenset[Permission]] = dict(BUILTIN_TIERS)

    def resolve(name: str, path: tuple[str, ...]) -> frozenset[Permission]:
        if name in resolved:
            return resolved[name]
        if name in path:
            cycle = " -> ".join((*path[path.index(name) :], name))
            raise AuthenticationConfigError(f"authentication.tiers: includes form a cycle ({cycle})")
        tier = custom[name]
        permissions = frozenset(Permission(permission) for permission in tier.permissions)
        for included in tier.includes:
            permissions |= resolve(included, (*path, name))
        resolved[name] = permissions
        return permissions

    for name in custom:
        resolve(name, ())
    return resolved


def _refuse_repeats(tokens: list[NamedToken]) -> None:
    for index, token in enumerate(tokens):
        for earlier in tokens[:index]:
            if earlier.id == token.id:
                raise AuthenticationConfigError(f"authentication.tokens: the id {token.id} is used twice")
            if hmac.compare_digest(earlier.secret.encode(), token.secret.encode()):
                raise AuthenticationConfigError(
                    f"authentication.tokens: {earlier.id} and {token.id} have the same secret"
                )


_loaded: tuple[ChapSettings, Authentication] | None = None


def get_authentication() -> Authentication:
    """The authentication for the current settings, resolved once per settings object."""
    global _loaded
    settings = get_settings()
    if _loaded is None or _loaded[0] is not settings:
        _loaded = (settings, load_authentication(settings))
    return _loaded[1]


def secret_matches(presented: str | None, expected: str) -> bool:
    """Timing-safe comparison of a presented secret against a configured one."""
    if not presented:
        return False
    # compare_digest on str raises TypeError for non-ascii, so compare encoded bytes.
    return hmac.compare_digest(presented.encode(), expected.encode())


def _matching_token(secret: str, tokens: tuple[NamedToken, ...]) -> NamedToken | None:
    # Every token is compared, so the time taken does not say which one came close.
    matches = [token for token in tokens if secret_matches(secret, token.secret)]
    return matches[0] if matches else None


def _from_authorization(header: str, tokens: tuple[NamedToken, ...]) -> Principal | None:
    scheme, _, credential = header.partition(" ")
    credential = credential.strip()
    token_id: str | None = None
    if scheme.lower() == "bearer":
        secret = credential.removeprefix("basic//")
    elif scheme.lower() == "apitoken":
        secret = credential
    elif scheme.lower() == "basic":
        try:
            token_id, _, secret = base64.b64decode(credential, validate=True).decode().partition(":")
        # binascii.Error, UnicodeDecodeError and a non-ASCII header are all ValueErrors.
        except ValueError:
            return None
    else:
        return None
    token = _matching_token(secret, tokens)
    if token is None or (token_id is not None and token.id != token_id):
        return None
    return token.principal


def authenticate(
    authorization: str | None, service_key: str | None, path: str, authentication: Authentication
) -> Principal | None:
    """The principal the credentials name: anonymous without any, None when none of them is valid.

    ``Authorization`` is tried first, then ``X-Service-Key``, so a client sending both is let in
    by whichever is valid, as before named tokens.
    """
    if not authorization and not service_key:
        return ANONYMOUS
    if authorization:
        principal = _from_authorization(authorization, authentication.tokens)
        if principal is not None:
            return principal
    if service_key:
        token = _matching_token(service_key, authentication.tokens)
        if token is not None:
            return token.principal
        key = authentication.registration_key
        if key is not None and path.startswith(SERVICE_REGISTRY_PREFIX) and secret_matches(service_key, key):
            return SERVICE_REGISTRY
    return None
