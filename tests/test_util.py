"""Unit tests for chap_core.util helpers."""

import re

from chap_core.util import generate_run_name, generate_short_id


def test_generate_short_id_default_is_8_hex_chars():
    sid = generate_short_id()
    assert len(sid) == 8
    assert all(c in "0123456789abcdef" for c in sid)


def test_generate_short_id_respects_length():
    assert len(generate_short_id(4)) == 4
    assert len(generate_short_id(12)) == 12


def test_generate_short_id_is_random():
    ids = {generate_short_id() for _ in range(100)}
    # 8 hex chars = 4 bytes; 100 draws colliding is astronomically unlikely.
    assert len(ids) == 100


def test_generate_run_name_format():
    name = generate_run_name()
    # <YYYY-MM-DD_HH-MM-SS>_<8 hex chars>, e.g. 2026-05-28_16-04-37_0e5fe728
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_[0-9a-f]{8}", name)


def test_generate_run_name_is_unique():
    # Same-second runs must differ thanks to the random suffix.
    assert generate_run_name() != generate_run_name()
