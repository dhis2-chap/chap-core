"""Unit tests for the WHO C-SUM threshold strategy.

Golden population-SD numbers match CAPS ``statistics.ts`` (divide by N, not N-1).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from chap_core.assessment.thresholds.c_sum import (
    _line_value,
    _population_std,
    _shift_season,
    collect_window_values,
    compute_c_sum_thresholds,
)
from chap_core.assessment.thresholds.params import CSumParams


def test_shift_season_wraps_year_boundaries():
    assert _shift_season(2022, 1, -1, 12) == (2021, 12)
    assert _shift_season(2022, 12, 1, 12) == (2023, 1)
    assert _shift_season(2022, 6, 0, 12) == (2022, 6)


def test_population_std_matches_caps_divide_by_n():
    window = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    mean = sum(window) / len(window)
    expected = math.sqrt(sum((v - mean) ** 2 for v in window) / len(window))
    assert _population_std(window) == pytest.approx(expected)
    # Distinct from sample SD (ddof=1).
    assert _population_std(window) != pytest.approx(float(np.std(window, ddof=1)))


def test_population_std_edge_cases():
    assert math.isnan(_population_std([]))
    assert _population_std([42.0]) == 0.0


def test_line_value_golden_caps_window():
    """CAPS statistics.test.ts window [10..60]: mean=35, pop SD≈17.078, upper≈68.473."""
    window = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    assert _line_value("mean", window) == 35.0
    assert _line_value("sd", window) == pytest.approx(17.07825127659933)
    assert _line_value("mean_plus_1_96_sd", window) == pytest.approx(68.47337250213468)
    assert math.isnan(_line_value("mean", []))


def test_collect_window_values_prev_current_next():
    lookup = {
        ("A", 2021, 12): 10.0,
        ("A", 2022, 1): 20.0,
        ("A", 2022, 2): 30.0,
    }
    assert collect_window_values(lookup, "A", 2022, 1, 12) == [10.0, 20.0, 30.0]


def test_compute_c_sum_thresholds_pools_three_period_window():
    """June threshold pools May/June/July across two baseline years."""
    rows = []
    # Year 2021: May=10, June=20, July=30
    rows.extend(
        [
            {"location": "A", "time_period": "2021-05", "disease_cases": 10.0},
            {"location": "A", "time_period": "2021-06", "disease_cases": 20.0},
            {"location": "A", "time_period": "2021-07", "disease_cases": 30.0},
        ]
    )
    # Year 2022: May=40, June=50, July=60
    rows.extend(
        [
            {"location": "A", "time_period": "2022-05", "disease_cases": 40.0},
            {"location": "A", "time_period": "2022-06", "disease_cases": 50.0},
            {"location": "A", "time_period": "2022-07", "disease_cases": 60.0},
        ]
    )
    # Fill remaining months so season detection stays monthly and years are complete enough
    # for callers that use filter_to_baseline; this unit test calls compute directly.
    for year in (2021, 2022):
        for month in range(1, 13):
            if month in (5, 6, 7):
                continue
            rows.append({"location": "A", "time_period": f"{year}-{month:02d}", "disease_cases": 0.0})

    df = pd.DataFrame(rows)
    result = compute_c_sum_thresholds(df, ["mean", "sd", "mean_plus_1_96_sd"])
    june = result[result["month"] == 6].set_index("line")["threshold"]
    window = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    assert june[0] == pytest.approx(_line_value("mean", window))
    assert june[1] == pytest.approx(_line_value("sd", window))
    assert june[2] == pytest.approx(_line_value("mean_plus_1_96_sd", window))


def test_compute_c_sum_omits_sparse_windows():
    """Fewer than three pooled values → no row (endpoint fills null)."""
    df = pd.DataFrame(
        [
            {"location": "A", "time_period": "2021-06", "disease_cases": 10.0},
            {"location": "A", "time_period": "2022-06", "disease_cases": 20.0},
        ]
    )
    result = compute_c_sum_thresholds(df, ["mean"])
    assert result.empty


def test_compute_c_sum_uses_lookup_for_year_boundary():
    """January of first baseline year can resolve prior December from lookup_observations."""
    baseline = pd.DataFrame(
        [
            {"location": "A", "time_period": "2022-01", "disease_cases": 20.0},
            {"location": "A", "time_period": "2022-02", "disease_cases": 30.0},
        ]
    )
    full = pd.concat(
        [
            pd.DataFrame([{"location": "A", "time_period": "2021-12", "disease_cases": 10.0}]),
            baseline,
        ],
        ignore_index=True,
    )
    result = compute_c_sum_thresholds(baseline, ["mean"], lookup_observations=full)
    jan = result[result["month"] == 1]
    assert len(jan) == 1
    assert jan.iloc[0]["threshold"] == pytest.approx(20.0)  # mean of [10, 20, 30]


def test_c_sum_params_defaults_and_lines_order():
    params = CSumParams(type="c_sum")
    assert params.lines == ["mean_plus_1_96_sd"]
    assert CSumParams(type="c_sum", statistic=["mean", "sd"]).lines == ["mean", "sd"]


def test_c_sum_params_type_required():
    with pytest.raises(ValueError):
        CSumParams()  # type: ignore[call-arg]


def test_c_sum_params_validation():
    with pytest.raises(ValueError):
        CSumParams(type="c_sum", statistic=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CSumParams(type="c_sum", baseline_years=0)
    with pytest.raises(ValueError):
        CSumParams(type="c_sum", statistic="not_a_line")  # type: ignore[arg-type]
