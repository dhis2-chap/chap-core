"""The permissions routes need, and the built-in tiers that grant them.

A permission is ``action:resource``. The set is fixed here; a deployment combines permissions
into custom tiers in ``chap-server.yaml`` but cannot add one.
"""

from enum import StrEnum


class Permission(StrEnum):
    READ_DATASETS = "read:datasets"
    WRITE_DATASETS = "write:datasets"
    DELETE_DATASETS = "delete:datasets"

    READ_MODELS = "read:models"
    WRITE_MODELS = "write:models"
    DELETE_MODELS = "delete:models"

    READ_BACKTESTS = "read:backtests"
    RUN_BACKTESTS = "run:backtests"
    WRITE_BACKTESTS = "write:backtests"
    DELETE_BACKTESTS = "delete:backtests"

    READ_PREDICTIONS = "read:predictions"
    RUN_PREDICTIONS = "run:predictions"
    DELETE_PREDICTIONS = "delete:predictions"

    READ_PREDICTION_SETUPS = "read:prediction-setups"
    WRITE_PREDICTION_SETUPS = "write:prediction-setups"
    DELETE_PREDICTION_SETUPS = "delete:prediction-setups"

    READ_ALERTS = "read:alerts"
    WRITE_ALERTS = "write:alerts"
    APPROVE_ALERTS = "approve:alerts"
    DELETE_ALERTS = "delete:alerts"

    # Job logs and requests can hold raw data, so reading jobs is not part of the read tier.
    READ_JOBS = "read:jobs"
    # Cancelling and forgetting jobs.
    WRITE_JOBS = "write:jobs"

    # The installation's defaults and the fixed lists clients choose from, such as plot types.
    READ_SETTINGS = "read:settings"
    WRITE_SETTINGS = "write:settings"
    # The on-demand probe that round-trips the broker and the database.
    READ_SYSTEM = "read:system"

    READ_SERVICES = "read:services"
    # The proxy to a registered chapkit service.
    CALL_SERVICES = "call:services"
    REGISTER_SERVICES = "register:services"


READ: frozenset[Permission] = frozenset(
    permission
    for permission in Permission
    if permission.startswith("read:") and permission not in (Permission.READ_JOBS, Permission.READ_SYSTEM)
)

AUTHENTICATED: frozenset[Permission] = READ | {
    Permission.READ_JOBS,
    Permission.WRITE_JOBS,
    Permission.READ_SYSTEM,
    Permission.WRITE_DATASETS,
    Permission.RUN_BACKTESTS,
    Permission.RUN_PREDICTIONS,
    Permission.WRITE_PREDICTION_SETUPS,
    Permission.CALL_SERVICES,
}

OPERATOR: frozenset[Permission] = frozenset(Permission)

# Given to callers without a token while authentication is on. They reach only the public routes.
PUBLIC: frozenset[Permission] = frozenset()

BUILTIN_TIERS: dict[str, frozenset[Permission]] = {
    "public": PUBLIC,
    "read": READ,
    "authenticated": AUTHENTICATED,
    "operator": OPERATOR,
}
