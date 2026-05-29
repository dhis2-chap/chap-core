"""Unit tests for chap_core.util helpers."""

from chap_core.util import generate_short_id


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
