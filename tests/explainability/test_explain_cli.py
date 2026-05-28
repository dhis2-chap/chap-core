"""Unit tests for the explain-lime CLI location resolution helper."""

import pytest

from chap_core.cli_endpoints.explain import _resolve_locations

AVAILABLE = ["Bokeo", "Savannakhet", "Vientiane[prefecture]"]


class TestResolveLocations:
    def test_single_requested(self):
        assert _resolve_locations(["Bokeo"], False, AVAILABLE) == ["Bokeo"]

    def test_multiple_preserve_order(self):
        assert _resolve_locations(["Savannakhet", "Bokeo"], False, AVAILABLE) == ["Savannakhet", "Bokeo"]

    def test_all_locations_returns_everything(self):
        assert _resolve_locations(None, True, AVAILABLE) == AVAILABLE

    def test_none_given_errors(self):
        with pytest.raises(ValueError, match="No location given"):
            _resolve_locations(None, False, AVAILABLE)

    def test_unknown_location_errors_and_lists_valid_options(self):
        with pytest.raises(ValueError, match="Unknown location") as exc:
            _resolve_locations(["Nowhere"], False, AVAILABLE)
        # The error should help the user by listing valid options.
        assert "Bokeo" in str(exc.value)

    def test_all_and_explicit_locations_conflict_errors(self):
        with pytest.raises(ValueError, match="not both"):
            _resolve_locations(["Bokeo"], True, AVAILABLE)
