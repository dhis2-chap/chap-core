import pytest

from chap_core.cli_endpoints._common import resolve_period_arg
from chap_core.time_period import Month, TimePeriod, Year

_PERIODS = [TimePeriod.parse(p) for p in ["2022-01", "2022-02", "2022-03"]]


def test_relative_last_period():
    assert resolve_period_arg("-1", _PERIODS, "arg") == Month(2022, 3)


def test_relative_second_to_last_period():
    assert resolve_period_arg("-2", _PERIODS, "arg") == Month(2022, 2)


def test_relative_first_period():
    assert resolve_period_arg("+1", _PERIODS, "arg") == Month(2022, 1)


def test_relative_second_period():
    assert resolve_period_arg("+2", _PERIODS, "arg") == Month(2022, 2)


def test_relative_index_out_of_range():
    with pytest.raises(ValueError, match="only 3 distinct periods"):
        resolve_period_arg("+10", _PERIODS, "arg")


def test_relative_index_zero_invalid():
    with pytest.raises(ValueError, match="must be at least 1"):
        resolve_period_arg("+0", _PERIODS, "arg")


def test_relative_index_negative_zero_invalid():
    with pytest.raises(ValueError, match="must be at least 1"):
        resolve_period_arg("-0", _PERIODS, "arg")


def test_exact_period_string_unaffected():
    assert resolve_period_arg("2022-02", _PERIODS, "arg") == Month(2022, 2)


def test_bare_year_string_treated_as_exact_period_not_relative_index():
    assert resolve_period_arg("2023", [TimePeriod.parse("2023")], "arg") == Year(2023)


def test_logs_resolved_period(caplog):
    with caplog.at_level("INFO"):
        resolve_period_arg("-1", _PERIODS, "split_period")
    assert "split_period" in caplog.text
    assert "resolved relative period '-1'" in caplog.text
