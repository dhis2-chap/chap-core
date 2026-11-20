import re

from chap_core.auth.permissions import AUTHENTICATED, BUILTIN_TIERS, OPERATOR, READ, Permission

ACTIONS = ("read", "export", "write", "delete", "run", "approve", "call", "register")


def test_every_permission_is_an_action_on_a_resource():
    pattern = re.compile(rf"^({'|'.join(ACTIONS)}):[a-z][a-z-]*$")

    assert [permission for permission in Permission if not pattern.match(permission)] == []


def test_each_built_in_tier_includes_the_one_below():
    assert READ < AUTHENTICATED < OPERATOR


def test_the_read_tier_only_reads_and_leaves_out_jobs():
    assert all(permission.startswith("read:") for permission in READ)
    assert Permission.READ_JOBS not in READ


def test_only_the_operator_tier_deletes():
    deleting = {permission for permission in Permission if permission.startswith("delete:")}

    assert {name for name, permissions in BUILTIN_TIERS.items() if permissions & deleting} == {"operator"}


def test_the_read_tier_cannot_export_a_datasets_observations():
    assert Permission.EXPORT_DATASETS not in READ
    assert Permission.EXPORT_DATASETS in AUTHENTICATED
