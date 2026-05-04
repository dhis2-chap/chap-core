"""Tests for the surrogate cache size override."""

import importlib

import pytest


def test_cache_max_respects_env(monkeypatch):
    monkeypatch.setenv("CHAP_SURROGATE_CACHE_MAX", "3")
    from chap_core.xai.surrogate import cache as cache_mod

    importlib.reload(cache_mod)
    try:
        for i in range(5):
            cache_mod.put_cached_surrogate((i,), object())
        # Only the last 3 keys remain (FIFO eviction).
        assert cache_mod.get_cached_surrogate((0,)) is None
        assert cache_mod.get_cached_surrogate((1,)) is None
        assert cache_mod.get_cached_surrogate((2,)) is not None
        assert cache_mod.get_cached_surrogate((4,)) is not None
    finally:
        importlib.reload(cache_mod)
