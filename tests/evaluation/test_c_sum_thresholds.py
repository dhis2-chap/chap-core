"""Tests for the WHO C-SUM threshold computation.

The golden window matches CAPS ``statistics.ts``: values 10..60 pooled give mean 35 and
a population SD (divide by N, not N-1) of about 17.078.
"""

from __future__ import annotations

import pandas as pd
import pytest

from chap_core.assessment.thresholds.c_sum import compute_c_sum_thresholds


def _frame(rows: dict[str, float], location: str = "A") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "location": location,
            "time_period": list(rows),
            "disease_cases": list(rows.values()),
        }
    )


def _threshold(result: pd.DataFrame, season_name: str, season: int, line: int = 0) -> float:
    row = result[(result[season_name] == season) & (result["line"] == line)]
    return float(row["threshold"].iloc[0])


GOLDEN = _frame(
    {
        "2020-05": 10.0,
        "2020-06": 20.0,
        "2020-07": 30.0,
        "2021-05": 40.0,
        "2021-06": 50.0,
        "2021-07": 60.0,
    }
)


def test_golden_caps_window():
    result = compute_c_sum_thresholds(GOLDEN, [0.0, 1.96])
    assert _threshold(result, "month", 6, line=0) == pytest.approx(35.0)
    assert _threshold(result, "month", 6, line=1) == pytest.approx(68.47337250213468)


def test_window_wraps_the_year_boundary():
    data = _frame({"2020-12": 10.0, "2021-01": 20.0, "2021-02": 30.0})
    result = compute_c_sum_thresholds(data, [0.0])
    assert _threshold(result, "month", 1) == pytest.approx(20.0)


def test_missing_current_period_drops_that_year_only():
    """A missing June 2021 must not leave a partial window from May and July 2021."""
    data = GOLDEN[GOLDEN["time_period"] != "2021-06"]
    result = compute_c_sum_thresholds(data, [0.0])
    assert _threshold(result, "month", 6) == pytest.approx(20.0)


def test_isolated_values_without_neighbours_give_no_threshold():
    data = _frame({"2019-06": 10.0, "2020-06": 20.0, "2021-06": 30.0})
    assert compute_c_sum_thresholds(data, [0.0]).empty


def test_nan_cases_are_treated_as_missing():
    data = GOLDEN.copy()
    data.loc[data["time_period"] == "2021-06", "disease_cases"] = float("nan")
    result = compute_c_sum_thresholds(data, [0.0])
    assert _threshold(result, "month", 6) == pytest.approx(20.0)


def test_week_53_is_a_neighbour_of_week_52_and_week_1():
    """2020 has 53 ISO weeks, so week 1 of 2021 follows week 53, not week 52."""
    data = _frame({"2020-W52": 10.0, "2020-W53": 20.0, "2021-W01": 30.0})
    result = compute_c_sum_thresholds(data, [0.0])
    assert _threshold(result, "week", 53) == pytest.approx(20.0)
    assert 1 not in set(result["week"])


def test_locations_are_pooled_separately():
    other = _frame({"2020-05": 100.0, "2020-06": 100.0, "2020-07": 100.0}, location="B")
    result = compute_c_sum_thresholds(pd.concat([GOLDEN, other]), [0.0])
    assert _threshold(result[result["location"] == "B"], "month", 6) == pytest.approx(100.0)
    assert _threshold(result[result["location"] == "A"], "month", 6) == pytest.approx(35.0)
