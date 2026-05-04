"""Tests for covariate_fallback and forecast_matching period logic."""

from __future__ import annotations

import math
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chap_core.xai.covariate_fallback import (
    covariate_period_candidates,
    period_df_for_forecast,
    resolve_covariate_row,
    target_signature,
    year_month_from_any,
)
from chap_core.xai.forecast_matching import find_forecast_row_index


# ---------------------------------------------------------------------------
# year_month_from_any
# ---------------------------------------------------------------------------


class TestYearMonthFromAny:
    def test_none_returns_none(self):
        assert year_month_from_any(None) is None

    def test_pd_period(self):
        assert year_month_from_any(pd.Period("2024-05", freq="M")) == (2024, 5)

    def test_pd_timestamp(self):
        assert year_month_from_any(pd.Timestamp("2024-05-15")) == (2024, 5)

    def test_date_object(self):
        assert year_month_from_any(date(2024, 5, 15)) == (2024, 5)

    def test_int_yyyymm(self):
        assert year_month_from_any(202405) == (2024, 5)

    def test_float_yyyymm(self):
        assert year_month_from_any(202405.0) == (2024, 5)

    def test_nan_float_returns_none(self):
        assert year_month_from_any(float("nan")) is None

    def test_string_6digit(self):
        assert year_month_from_any("202405") == (2024, 5)

    def test_string_yyyymm_with_trailing_dot_zero(self):
        assert year_month_from_any("202405.0") == (2024, 5)

    def test_string_iso_month(self):
        assert year_month_from_any("2024-05") == (2024, 5)

    def test_string_iso_date(self):
        assert year_month_from_any("2024-05-01") == (2024, 5)

    def test_string_nan_returns_none(self):
        assert year_month_from_any("nan") is None

    def test_string_nat_returns_none(self):
        assert year_month_from_any("NaT") is None

    def test_numpy_scalar(self):
        assert year_month_from_any(np.int64(202405)) == (2024, 5)

    def test_empty_string_returns_none(self):
        assert year_month_from_any("") is None


# ---------------------------------------------------------------------------
# covariate_period_candidates
# ---------------------------------------------------------------------------


class TestCovariatePeriodCandidates:
    def test_plain_month_returns_self(self):
        assert covariate_period_candidates("202405") == ["202405"]

    def test_horizon_step_returns_shifted_then_base(self):
        # "202405_3" → shifted = May 2024 + 3 months = Aug 2024 = "202408"
        cands = covariate_period_candidates("202405_3")
        assert cands[0] == "202405_3"
        assert "202408" in cands
        assert "202405" in cands
        assert cands.index("202408") < cands.index("202405")

    def test_horizon_step_1(self):
        cands = covariate_period_candidates("202405_1")
        assert "202406" in cands

    def test_no_duplicates(self):
        cands = covariate_period_candidates("202405_3")
        assert len(cands) == len(set(cands))

    def test_non_digit_suffix_treated_as_base(self):
        # suffix is not a digit → just [fp, base]
        cands = covariate_period_candidates("202405_abc")
        assert cands[0] == "202405_abc"
        assert "202405" in cands


# ---------------------------------------------------------------------------
# period_df_for_forecast
# ---------------------------------------------------------------------------


@pytest.fixture()
def loc_df_monthly():
    return pd.DataFrame(
        {
            "period": ["202403", "202404", "202405", "202406"],
            "rainfall": [10.0, 20.0, 30.0, 40.0],
        }
    )


class TestPeriodDfForForecast:
    def test_empty_df_returns_empty(self):
        result = period_df_for_forecast(pd.DataFrame(), "period", "202405")
        assert result.empty

    def test_exact_string_match(self, loc_df_monthly):
        result = period_df_for_forecast(loc_df_monthly, "period", "202405")
        assert len(result) == 1
        assert result["rainfall"].iloc[0] == 30.0

    def test_year_month_match_via_iso_string(self):
        df = pd.DataFrame({"period": ["2024-03", "2024-04", "2024-05"], "v": [1, 2, 3]})
        result = period_df_for_forecast(df, "period", "202405")
        assert len(result) == 1
        assert result["v"].iloc[0] == 3

    def test_horizon_step_falls_back_to_shifted(self, loc_df_monthly):
        # "202403_3" → shifted = Jun 2024 = "202406"; direct "202403_3" won't match
        result = period_df_for_forecast(loc_df_monthly, "period", "202403_3")
        assert len(result) == 1
        assert result["period"].iloc[0] == "202406"

    def test_no_match_returns_empty(self, loc_df_monthly):
        result = period_df_for_forecast(loc_df_monthly, "period", "202501")
        assert result.empty

    def test_missing_period_col_returns_empty(self, loc_df_monthly):
        result = period_df_for_forecast(loc_df_monthly, "no_such_col", "202405")
        assert result.empty


# ---------------------------------------------------------------------------
# target_signature
# ---------------------------------------------------------------------------


class TestTargetSignature:
    def test_plain_month_string(self):
        assert target_signature("202405") == ("month", 2024, 5)

    def test_horizon_step_3(self):
        # May 2024 + 3 = August 2024
        assert target_signature("202405_3") == ("month", 2024, 8)

    def test_horizon_step_1(self):
        assert target_signature("202405_1") == ("month", 2024, 6)

    def test_horizon_step_wraps_year(self):
        # November 2024 + 3 = February 2025
        assert target_signature("202411_3") == ("month", 2025, 2)

    def test_invalid_returns_none(self):
        assert target_signature("not-a-period") is None


# ---------------------------------------------------------------------------
# resolve_covariate_row
# ---------------------------------------------------------------------------


def _make_loc_df(periods, values):
    return pd.DataFrame({"period": periods, "rain": values})


def _make_global_df():
    return pd.DataFrame({"period": ["202301"], "rain": [99.0]})


class TestResolveCovariateRow:
    def test_empty_loc_df_uses_last_global_row(self):
        global_df = _make_global_df()
        row, prov = resolve_covariate_row(pd.DataFrame(), "period", ["rain"], "202405", "loc_A", global_df)
        assert prov["source"] == "last_available_row"
        assert row["rain"] == 99.0

    def test_exact_period_match(self):
        loc_df = _make_loc_df(["202403", "202404", "202405"], [10.0, 20.0, 30.0])
        row, prov = resolve_covariate_row(loc_df, "period", ["rain"], "202405", "loc_A", _make_global_df())
        assert prov["source"] == "dataset_match"
        assert prov["matchedPeriod"] == "202405"
        assert row["rain"] == 30.0

    def test_historical_same_month_fallback(self):
        # Forecast for May 2025 but dataset only has up to Dec 2024 → prior-year May
        loc_df = _make_loc_df(["202305", "202405"], [5.0, 15.0])
        row, prov = resolve_covariate_row(loc_df, "period", ["rain"], "202505", "loc_A", _make_global_df())
        assert prov["source"] == "historical_same_month_mean"
        assert prov["calendarMonth"] == 5
        assert math.isclose(row["rain"], 10.0)  # mean of 5 and 15

    def test_historical_fallback_uses_only_prior_years(self):
        # Forecast for May 2024; dataset has May 2022, May 2023, May 2025
        loc_df = _make_loc_df(["202205", "202305", "202505"], [2.0, 4.0, 100.0])
        row, prov = resolve_covariate_row(loc_df, "period", ["rain"], "202405", "loc_A", _make_global_df())
        assert prov["source"] == "historical_same_month_mean"
        assert 2025 not in prov["yearsUsed"]
        assert math.isclose(row["rain"], 3.0)  # mean of 2 and 4

    def test_last_row_fallback_when_no_historical(self):
        # Forecast period matches no month at all
        loc_df = _make_loc_df(["202306", "202307"], [7.0, 8.0])
        row, prov = resolve_covariate_row(loc_df, "period", ["rain"], "202405_5", "loc_A", _make_global_df())
        # "202405_5" → target = Oct 2024; no Oct rows → last row fallback
        assert prov["source"] == "last_available_row"
        assert row["rain"] == 8.0

    def test_missing_feature_returns_nan(self):
        loc_df = _make_loc_df(["202405"], [30.0])
        row, _ = resolve_covariate_row(loc_df, "period", ["rain", "temp"], "202405", "loc_A", _make_global_df())
        assert not math.isnan(row["rain"])
        assert math.isnan(row["temp"])

    def test_no_period_col_uses_last_row(self):
        loc_df = _make_loc_df(["202405"], [30.0])
        row, prov = resolve_covariate_row(loc_df, "", ["rain"], "202405", "loc_A", _make_global_df())
        assert prov["source"] == "last_available_row"


# ---------------------------------------------------------------------------
# find_forecast_row_index
# ---------------------------------------------------------------------------


def _forecast(org_unit: str, period: str) -> SimpleNamespace:
    return SimpleNamespace(org_unit=org_unit, period=period)


class TestFindForecastRowIndex:
    def test_exact_match(self):
        forecasts = [_forecast("A", "202405"), _forecast("A", "202406"), _forecast("B", "202405")]
        assert find_forecast_row_index(forecasts, "A", "202405") == 0
        assert find_forecast_row_index(forecasts, "A", "202406") == 1
        assert find_forecast_row_index(forecasts, "B", "202405") == 2

    def test_dash_underscore_normalization(self):
        forecasts = [_forecast("A", "202405_1"), _forecast("A", "202405_2")]
        # Request uses dashes; stored uses underscores
        assert find_forecast_row_index(forecasts, "A", "202405-1") == 0
        assert find_forecast_row_index(forecasts, "A", "202405-2") == 1

    def test_org_unit_not_found_returns_none(self):
        forecasts = [_forecast("A", "202405")]
        assert find_forecast_row_index(forecasts, "B", "202405") is None

    def test_period_not_found_for_org_unit_returns_none(self):
        forecasts = [_forecast("A", "202405"), _forecast("A", "202406")]
        assert find_forecast_row_index(forecasts, "A", "202412") is None

    def test_horizon_step_matches_calendar_month(self):
        # Forecasts stored as plain calendar periods; request uses "202405_3" → target Aug 2024
        forecasts = [
            _forecast("A", "202406"),
            _forecast("A", "202407"),
            _forecast("A", "202408"),
        ]
        assert find_forecast_row_index(forecasts, "A", "202405_3") == 2

    def test_single_org_unit_entry_returned_as_fallback(self):
        forecasts = [_forecast("A", "202405"), _forecast("B", "202409")]
        # Only one forecast for "B"; period doesn't match literally but it's the only one
        assert find_forecast_row_index(forecasts, "B", "202409") == 1

    def test_empty_forecast_list_returns_none(self):
        assert find_forecast_row_index([], "A", "202405") is None
