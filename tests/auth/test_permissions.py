import re

from chap_core.auth.permissions import ANALYST, BUILTIN_TIERS, ADMIN, VIEWER, Permission

ACTIONS = ("read", "export", "write", "delete", "run", "approve", "call", "register")


def test_every_permission_is_an_action_on_a_resource():
    pattern = re.compile(rf"^({'|'.join(ACTIONS)}):[a-z][a-z-]*$")

    assert [permission for permission in Permission if not pattern.match(permission)] == []


def test_each_built_in_tier_includes_the_one_below():
    assert VIEWER < ANALYST < ADMIN


def test_the_read_tier_only_reads_and_leaves_out_jobs():
    assert all(permission.startswith("read:") for permission in VIEWER)
    assert Permission.READ_JOBS not in VIEWER


def test_only_the_operator_tier_deletes():
    deleting = {permission for permission in Permission if permission.startswith("delete:")}

    assert {name for name, permissions in BUILTIN_TIERS.items() if permissions & deleting} == {"admin"}


def test_the_read_tier_cannot_export_a_datasets_observations():
    assert Permission.EXPORT_DATASETS not in VIEWER
    assert Permission.EXPORT_DATASETS in ANALYST
