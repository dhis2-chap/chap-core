import math
from unittest.mock import patch

import pandas as pd
import pytest

from chap_core.cli_endpoints.build_counterfactual import build_counterfactual_cmd
from chap_core.cli_endpoints.build_counterfactual import FeatureTransformations

_LOCATIONS = ["A"]
_PERIODS = ["2022-01", "2022-02", "2022-03"]


# --- validate_expression ---


def test_validate_expression_addition():
    FeatureTransformations.validate_expression("x+10")


def test_validate_expression_subtraction():
    FeatureTransformations.validate_expression("x-30")


def test_validate_expression_multiplication():
    FeatureTransformations.validate_expression("x*0.1")


def test_validate_expression_division():
    FeatureTransformations.validate_expression("x/2")


def test_validate_expression_composite():
    FeatureTransformations.validate_expression("x*0+1")


def test_validate_expression_inverted():
    FeatureTransformations.validate_expression("1-x")


def test_validate_expression_abs():
    FeatureTransformations.validate_expression("abs(x)")


def test_validate_expression_round():
    FeatureTransformations.validate_expression("round(x)")


def test_validate_expression_nested():
    FeatureTransformations.validate_expression("abs(x*0.1-5)")


def test_validate_expression_disallowed_name():
    with pytest.raises(ValueError, match="Disallowed name"):
        FeatureTransformations.validate_expression("y+1")


def test_validate_expression_disallowed_function():
    with pytest.raises(ValueError, match="Disallowed function"):
        FeatureTransformations.validate_expression("int(x)")


def test_validate_expression_string_constant():
    with pytest.raises(ValueError, match="Non-numeric constant"):
        FeatureTransformations.validate_expression("x+'a'")


def test_validate_expression_syntax_error():
    with pytest.raises(ValueError, match="Invalid expression"):
        FeatureTransformations.validate_expression("x +* 1")


def test_validate_expression_exponentiation():
    FeatureTransformations.validate_expression("x**2")


def test_validate_expression_disallowed_operator():
    with pytest.raises(ValueError):
        FeatureTransformations.validate_expression("x // 2")


# --- parse_transformations ---


def test_parse_transformations_single():
    assert FeatureTransformations.parse_transformations(["rainfall=x+10"]) == [("rainfall", "x+10")]


def testparse_transformations_multiple():
    result = FeatureTransformations.parse_transformations(["rainfall=x*0.01", "temperature=x-30"])
    assert result == [("rainfall", "x*0.01"), ("temperature", "x-30")]


def test_parse_transformations_splits_at_first_equals():
    # expression itself contains =  → only first = is the separator
    result = FeatureTransformations.parse_transformations(["col=x*0+1"])
    assert result == [("col", "x*0+1")]


def test_parse_transformations_missing_separator():
    with pytest.raises(ValueError, match="not in 'column=expression' format"):
        FeatureTransformations.parse_transformations(["rainfall"])


# --- apply_transformation ---


def test_apply_transformation_basic():
    s = pd.Series([1.0, 2.0, 3.0])
    result = FeatureTransformations.apply_transformation(s, "x+10")
    assert list(result) == [11.0, 12.0, 13.0]


def test_apply_transformation_nan_unchanged():
    s = pd.Series([1.0, float("nan"), 3.0])
    result = FeatureTransformations.apply_transformation(s, "x*2")
    assert result[0] == 2.0
    assert math.isnan(result[1])
    assert result[2] == 6.0


def test_apply_transformation_abs():
    s = pd.Series([-3.0, 2.0])
    result = FeatureTransformations.apply_transformation(s, "abs(x)")
    assert list(result) == [3.0, 2.0]


def test_apply_transformation_round():
    s = pd.Series([1.4, 2.6])
    result = FeatureTransformations.apply_transformation(s, "round(x)")
    assert list(result) == [1, 3]


def test_apply_transformation_one_minus_x():
    s = pd.Series([0.2, 0.8])
    result = FeatureTransformations.apply_transformation(s, "1-x")
    assert list(result) == pytest.approx([0.8, 0.2])


# --- build_counterfactual_cmd ---


def test_basic_addition(tmp_path, make_test_df):
    csv = tmp_path / "data.csv"
    make_test_df(_LOCATIONS, _PERIODS).to_csv(csv, index=False)
    build_counterfactual_cmd(csv, ["rainfall=x+10"])
    out = pd.read_csv(tmp_path / "data_cf.csv")
    assert list(out["rainfall"]) == [11.0, 11.0, 11.0]
    assert list(out["disease_cases"]) == [0.0, 0.0, 0.0]


def test_multiple_transformations(tmp_path, make_test_df):
    csv = tmp_path / "data.csv"
    make_test_df(_LOCATIONS, _PERIODS).to_csv(csv, index=False)
    build_counterfactual_cmd(csv, ["rainfall=x*2", "disease_cases=x+5"])
    out = pd.read_csv(tmp_path / "data_cf.csv")
    assert list(out["rainfall"]) == [2.0, 2.0, 2.0]
    assert list(out["disease_cases"]) == [5.0, 5.0, 5.0]


def test_missing_values_unchanged(tmp_path, make_test_df):
    df = make_test_df(_LOCATIONS, _PERIODS)
    df.loc[1, "rainfall"] = float("nan")
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    build_counterfactual_cmd(csv, ["rainfall=x+100"])
    out = pd.read_csv(tmp_path / "data_cf.csv")
    assert out["rainfall"][0] == 101.0
    assert math.isnan(out["rainfall"][1])
    assert out["rainfall"][2] == 101.0


def test_type_change_warning(tmp_path):
    df = pd.DataFrame(
        {
            "location": ["A"],
            "time_period": ["2022-01"],
            "rainfall": pd.array([10], dtype="int64"),
        }
    )
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    with patch("chap_core.cli_endpoints.build_counterfactual.logger") as mock_logger:
        build_counterfactual_cmd(csv, ["rainfall=x+0.5"])
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "changed type" in warning_msg


def test_start_time_period(tmp_path, make_test_df):
    csv = tmp_path / "data.csv"
    make_test_df(_LOCATIONS, _PERIODS).to_csv(csv, index=False)
    build_counterfactual_cmd(csv, ["rainfall=x+100"], start_time_period="2022-02")
    out = pd.read_csv(tmp_path / "data_cf.csv")
    assert out["rainfall"][0] == 1.0  # unchanged
    assert out["rainfall"][1] == 101.0  # modified
    assert out["rainfall"][2] == 101.0  # modified


def test_end_time_period(tmp_path, make_test_df):
    csv = tmp_path / "data.csv"
    make_test_df(_LOCATIONS, _PERIODS).to_csv(csv, index=False)
    build_counterfactual_cmd(csv, ["rainfall=x+100"], end_time_period="2022-02")
    out = pd.read_csv(tmp_path / "data_cf.csv")
    assert out["rainfall"][0] == 101.0  # modified
    assert out["rainfall"][1] == 101.0  # modified
    assert out["rainfall"][2] == 1.0  # unchanged


def test_time_period_range(tmp_path, make_test_df):
    csv = tmp_path / "data.csv"
    make_test_df(_LOCATIONS, _PERIODS).to_csv(csv, index=False)
    build_counterfactual_cmd(
        csv,
        ["rainfall=x+100"],
        start_time_period="2022-02",
        end_time_period="2022-02",
    )
    out = pd.read_csv(tmp_path / "data_cf.csv")
    assert out["rainfall"][0] == 1.0  # unchanged
    assert out["rainfall"][1] == 101.0  # modified
    assert out["rainfall"][2] == 1.0  # unchanged


# --- relative period indices ---


def test_start_time_period_relative_from_end(tmp_path, make_test_df):
    csv = tmp_path / "data.csv"
    make_test_df(_LOCATIONS, _PERIODS).to_csv(csv, index=False)
    build_counterfactual_cmd(csv, ["rainfall=x+100"], start_time_period="-2")
    out = pd.read_csv(tmp_path / "data_cf.csv")
    assert out["rainfall"][0] == 1.0  # unchanged
    assert out["rainfall"][1] == 101.0  # modified: 2nd-to-last period onward
    assert out["rainfall"][2] == 101.0  # modified


def test_end_time_period_relative_from_start(tmp_path, make_test_df):
    csv = tmp_path / "data.csv"
    make_test_df(_LOCATIONS, _PERIODS).to_csv(csv, index=False)
    build_counterfactual_cmd(csv, ["rainfall=x+100"], end_time_period="+2")
    out = pd.read_csv(tmp_path / "data_cf.csv")
    assert out["rainfall"][0] == 101.0  # modified
    assert out["rainfall"][1] == 101.0  # modified: up to 2nd period
    assert out["rainfall"][2] == 1.0  # unchanged


def test_relative_period_out_of_range(tmp_path, make_test_df):
    csv = tmp_path / "data.csv"
    make_test_df(_LOCATIONS, _PERIODS).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="only 3 distinct periods"):
        build_counterfactual_cmd(csv, ["rainfall=x+100"], start_time_period="-10")


def test_relative_period_zero_index_invalid(tmp_path, make_test_df):
    csv = tmp_path / "data.csv"
    make_test_df(_LOCATIONS, _PERIODS).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="must be at least 1"):
        build_counterfactual_cmd(csv, ["rainfall=x+100"], start_time_period="+0")


def test_unnamed_index_column_dropped_from_output(tmp_path, make_test_df):
    df = make_test_df(_LOCATIONS, _PERIODS)
    csv = tmp_path / "data.csv"
    df.to_csv(csv)  # no index=False: writes a leading unnamed index column, like an upstream artifact
    assert "Unnamed: 0" in pd.read_csv(csv).columns
    build_counterfactual_cmd(csv, ["rainfall=x+1"])
    out = pd.read_csv(tmp_path / "data_cf.csv")
    assert "Unnamed: 0" not in out.columns
    assert list(out.columns) == ["location", "time_period", "rainfall", "disease_cases"]


def test_default_output_name(tmp_path, make_test_df):
    csv = tmp_path / "my_data.csv"
    make_test_df(_LOCATIONS, _PERIODS).to_csv(csv, index=False)
    build_counterfactual_cmd(csv, ["rainfall=x*2"])
    assert (tmp_path / "my_data_cf.csv").exists()
    assert not (tmp_path / "my_data.csv_cf.csv").exists()


def test_custom_output_path(tmp_path, make_test_df):
    csv = tmp_path / "data.csv"
    out = tmp_path / "custom_output.csv"
    make_test_df(_LOCATIONS, _PERIODS).to_csv(csv, index=False)
    build_counterfactual_cmd(csv, ["rainfall=x*2"], output_csv=out)
    assert out.exists()
    assert not (tmp_path / "data_cf.csv").exists()


def test_validation_column_not_found(tmp_path, make_test_df):
    csv = tmp_path / "data.csv"
    make_test_df(_LOCATIONS, _PERIODS).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
        build_counterfactual_cmd(csv, ["nonexistent=x+1"])


def test_validation_invalid_expression(tmp_path, make_test_df):
    csv = tmp_path / "data.csv"
    make_test_df(_LOCATIONS, _PERIODS).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="Disallowed name"):
        build_counterfactual_cmd(csv, ["rainfall=y+1"])


def test_validation_column_name_contains_equals(tmp_path):
    df = pd.DataFrame({"location": ["A"], "time_period": ["2022-01"], "rain=fall": [1.0]})
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="must not contain '='"):
        build_counterfactual_cmd(csv, ["rainfall=x+1"])


def test_validation_missing_time_period_column(tmp_path):
    df = pd.DataFrame({"location": ["A"], "rainfall": [1.0]})
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="time_period"):
        build_counterfactual_cmd(csv, ["rainfall=x+1"], start_time_period="2022-01")


def test_validation_bad_format(tmp_path, make_test_df):
    csv = tmp_path / "data.csv"
    make_test_df(_LOCATIONS, _PERIODS).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="not in 'column=expression' format"):
        build_counterfactual_cmd(csv, ["rainfall"])


def test_abs_expression(tmp_path, make_test_df):
    csv = tmp_path / "data.csv"
    make_test_df(["A"], ["2022-01"], extra_col_val=-5.0).to_csv(csv, index=False)
    build_counterfactual_cmd(csv, ["rainfall=abs(x)"])
    out = pd.read_csv(tmp_path / "data_cf.csv")
    assert out["rainfall"][0] == 5.0


def test_round_expression(tmp_path, make_test_df):
    csv = tmp_path / "data.csv"
    make_test_df(["A"], ["2022-01"], extra_col_val=3.7).to_csv(csv, index=False)
    build_counterfactual_cmd(csv, ["rainfall=round(x)"])
    out = pd.read_csv(tmp_path / "data_cf.csv")
    assert out["rainfall"][0] == 4


def test_complex_expression(tmp_path, make_test_df):
    csv = tmp_path / "data.csv"
    make_test_df(_LOCATIONS, _PERIODS).to_csv(csv, index=False)
    build_counterfactual_cmd(csv, ["rainfall=x*0+1"])
    out = pd.read_csv(tmp_path / "data_cf.csv")
    assert list(out["rainfall"]) == [1.0, 1.0, 1.0]


def test_one_minus_x(tmp_path, make_test_df):
    csv = tmp_path / "data.csv"
    make_test_df(["A"], ["2022-01"], extra_col_val=0.3).to_csv(csv, index=False)
    build_counterfactual_cmd(csv, ["rainfall=1-x"])
    out = pd.read_csv(tmp_path / "data_cf.csv")
    assert out["rainfall"][0] == pytest.approx(0.7)


# --- seasonal_min / seasonal_max ---


def test_seasonal_min_same_month_previous_years(tmp_path, make_row_df):
    csv = tmp_path / "data.csv"
    make_row_df(
        [
            ("A", "2021-06", 40.0),
            ("A", "2022-06", 55.0),
            ("A", "2023-06", 999.0),
        ]
    ).to_csv(csv, index=False)
    build_counterfactual_cmd(csv, ["rainfall=seasonal_min"], start_time_period="2023-06")
    out = pd.read_csv(tmp_path / "data_cf.csv")
    assert out["rainfall"][2] == 40.0


def test_seasonal_max_same_week_previous_years(tmp_path, make_row_df):
    csv = tmp_path / "data.csv"
    make_row_df(
        [
            ("A", "2021-W10", 40.0),
            ("A", "2022-W10", 55.0),
            ("A", "2023-W10", 999.0),
        ]
    ).to_csv(csv, index=False)
    build_counterfactual_cmd(csv, ["rainfall=seasonal_max"], start_time_period="2023-W10")
    out = pd.read_csv(tmp_path / "data_cf.csv")
    assert out["rainfall"][2] == 55.0


def test_seasonal_ignores_out_of_season_rows(tmp_path, make_row_df):
    csv = tmp_path / "data.csv"
    make_row_df(
        [
            ("A", "2021-06", 40.0),
            ("A", "2022-06", 55.0),
            ("A", "2022-07", 1.0),
            ("A", "2023-06", 999.0),
        ]
    ).to_csv(csv, index=False)
    build_counterfactual_cmd(csv, ["rainfall=seasonal_min"], start_time_period="2023-06")
    out = pd.read_csv(tmp_path / "data_cf.csv")
    assert out["rainfall"][3] == 40.0  # not the out-of-season 1.0


def test_seasonal_per_location_isolation(tmp_path, make_row_df):
    csv = tmp_path / "data.csv"
    make_row_df(
        [
            ("A", "2022-06", 40.0),
            ("A", "2023-06", 999.0),
            ("B", "2022-06", 1.0),
            ("B", "2023-06", 999.0),
        ]
    ).to_csv(csv, index=False)
    build_counterfactual_cmd(csv, ["rainfall=seasonal_min"], start_time_period="2023-06")
    out = pd.read_csv(tmp_path / "data_cf.csv")
    assert out["rainfall"][1] == 40.0  # A's target, from A's history only
    assert out["rainfall"][3] == 1.0  # B's target, from B's history only


def test_seasonal_unsupported_period_type(tmp_path, make_row_df):
    csv = tmp_path / "data.csv"
    make_row_df([("A", "2022", 40.0), ("A", "2023", 999.0)]).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="require Month or Week"):
        build_counterfactual_cmd(csv, ["rainfall=seasonal_min"], start_time_period="2023")


def test_seasonal_missing_location_column(tmp_path, make_row_df):
    csv = tmp_path / "data.csv"
    make_row_df(
        [("2022-06", 40.0), ("2023-06", 999.0)],
        columns=("time_period", "rainfall"),
    ).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="Column 'location' not found"):
        build_counterfactual_cmd(csv, ["rainfall=seasonal_min"], start_time_period="2023-06")


def test_seasonal_requires_start_time_period(tmp_path, make_test_df):
    csv = tmp_path / "data.csv"
    make_test_df(_LOCATIONS, _PERIODS).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="--start-time-period is required"):
        build_counterfactual_cmd(csv, ["rainfall=seasonal_min"])


def test_seasonal_no_history_warns_and_skips(tmp_path, make_row_df):
    csv = tmp_path / "data.csv"
    make_row_df([("A", "2023-06", 999.0)]).to_csv(csv, index=False)
    with patch("chap_core.cli_endpoints.build_counterfactual.logger") as mock_logger:
        build_counterfactual_cmd(csv, ["rainfall=seasonal_min"], start_time_period="2023-06")
        mock_logger.warning.assert_called_once()
    out = pd.read_csv(tmp_path / "data_cf.csv")
    assert out["rainfall"][0] == 999.0


def test_seasonal_overrides_nan_target(tmp_path, make_row_df):
    csv = tmp_path / "data.csv"
    make_row_df(
        [
            ("A", "2021-06", 40.0),
            ("A", "2022-06", 55.0),
            ("A", "2023-06", float("nan")),
        ]
    ).to_csv(csv, index=False)
    build_counterfactual_cmd(csv, ["rainfall=seasonal_min"], start_time_period="2023-06")
    out = pd.read_csv(tmp_path / "data_cf.csv")
    assert out["rainfall"][2] == 40.0


# --- window_avg_min / window_avg_max ---

_WINDOW_AVG_ROWS = [
    ("A", "2021-01", 10.0),
    ("A", "2021-02", 20.0),
    ("A", "2021-03", 30.0),
    ("A", "2021-04", 5.0),
    ("A", "2021-05", 999.0),
    ("A", "2021-06", 999.0),
]


def test_window_avg_min_basic(tmp_path, make_row_df):
    csv = tmp_path / "data.csv"
    make_row_df(_WINDOW_AVG_ROWS).to_csv(csv, index=False)
    build_counterfactual_cmd(csv, ["rainfall=window_avg_min"], start_time_period="2021-05")
    out = pd.read_csv(tmp_path / "data_cf.csv")
    # windows [10,20] -> min 10, [30,5] -> min 5; avg = 7.5
    assert out["rainfall"][4] == pytest.approx(7.5)
    assert out["rainfall"][5] == pytest.approx(7.5)


def test_window_avg_max_basic(tmp_path, make_row_df):
    csv = tmp_path / "data.csv"
    make_row_df(_WINDOW_AVG_ROWS).to_csv(csv, index=False)
    build_counterfactual_cmd(csv, ["rainfall=window_avg_max"], start_time_period="2021-05")
    out = pd.read_csv(tmp_path / "data_cf.csv")
    # windows [10,20] -> max 20, [30,5] -> max 30; avg = 25
    assert out["rainfall"][4] == pytest.approx(25.0)
    assert out["rainfall"][5] == pytest.approx(25.0)


def test_window_avg_drops_partial_trailing_window(tmp_path, make_row_df):
    csv = tmp_path / "data.csv"
    make_row_df(
        [
            ("A", "2021-01", 10.0),
            ("A", "2021-02", 20.0),
            ("A", "2021-03", 1.0),
            ("A", "2021-04", 999.0),
            ("A", "2021-05", 999.0),
        ]
    ).to_csv(csv, index=False)
    build_counterfactual_cmd(csv, ["rainfall=window_avg_min"], start_time_period="2021-04")
    out = pd.read_csv(tmp_path / "data_cf.csv")
    # only one full window [10,20] -> min 10; the leftover 2021-03=1.0 is dropped, not averaged in
    assert out["rainfall"][3] == pytest.approx(10.0)
    assert out["rainfall"][4] == pytest.approx(10.0)


def test_window_avg_per_location_window_length(tmp_path, make_row_df):
    csv = tmp_path / "data.csv"
    make_row_df(
        [
            ("A", "2021-01", 100.0),
            ("A", "2021-02", 200.0),
            ("A", "2021-05", 999.0),
            ("B", "2021-01", 10.0),
            ("B", "2021-02", 20.0),
            ("B", "2021-03", 30.0),
            ("B", "2021-04", 40.0),
            ("B", "2021-05", 999.0),
            ("B", "2021-06", 999.0),
        ]
    ).to_csv(csv, index=False)
    build_counterfactual_cmd(csv, ["rainfall=window_avg_min"], start_time_period="2021-05")
    out = pd.read_csv(tmp_path / "data_cf.csv").set_index(["location", "time_period"])["rainfall"]
    # A: window length 1 -> windows [100],[200] -> avg(100,200) = 150
    assert out[("A", "2021-05")] == pytest.approx(150.0)
    # B: window length 2 -> windows [10,20]->10, [30,40]->30 -> avg = 20
    assert out[("B", "2021-05")] == pytest.approx(20.0)
    assert out[("B", "2021-06")] == pytest.approx(20.0)


def test_window_avg_gap_uses_only_present_periods_on_fixed_grid(tmp_path, make_row_df):
    csv = tmp_path / "data.csv"
    make_row_df(
        [
            ("A", "2021-01", 10.0),
            ("A", "2021-02", 20.0),
            # gap: no 2021-03
            ("A", "2021-04", 4.0),
            ("A", "2021-05", 5.0),
            ("A", "2021-06", 6.0),
            # active range (length 2)
            ("A", "2021-10", 999.0),
            ("A", "2021-11", 999.0),
        ]
    ).to_csv(csv, index=False)
    build_counterfactual_cmd(csv, ["rainfall=window_avg_min"], start_time_period="2021-10")
    out = pd.read_csv(tmp_path / "data_cf.csv")
    # fixed 2-period grid from 2021-01: [Jan,Feb] -> min(10,20)=10;
    # [Mar,Apr] -> only Apr present -> 4; [May,Jun] -> min(5,6)=5. avg(10, 4, 5)
    assert out["rainfall"][5] == pytest.approx(19 / 3)
    assert out["rainfall"][6] == pytest.approx(19 / 3)


def test_window_avg_skips_window_whose_span_is_entirely_gap(tmp_path, make_row_df):
    csv = tmp_path / "data.csv"
    make_row_df(
        [
            ("A", "2021-01", 10.0),
            ("A", "2021-02", 20.0),
            # gap: no 2021-03, 2021-04 -> the [Mar,Apr] grid window is empty and skipped
            ("A", "2021-05", 5.0),
            ("A", "2021-06", 50.0),
            ("A", "2021-10", 999.0),
            ("A", "2021-11", 999.0),
        ]
    ).to_csv(csv, index=False)
    build_counterfactual_cmd(csv, ["rainfall=window_avg_min"], start_time_period="2021-10")
    out = pd.read_csv(tmp_path / "data_cf.csv")
    # grid: [Jan,Feb] -> 10; [Mar,Apr] -> no present period, skipped; [May,Jun] -> 5. avg(10, 5) = 7.5
    assert out["rainfall"][4] == pytest.approx(7.5)
    assert out["rainfall"][5] == pytest.approx(7.5)


def test_mixed_arithmetic_and_seasonal_columns(tmp_path, make_row_df):
    csv = tmp_path / "data.csv"
    make_row_df(
        [
            ("A", "2021-06", 40.0, 10.0),
            ("A", "2022-06", 55.0, 10.0),
            ("A", "2023-06", 999.0, 10.0),
        ],
        columns=("location", "time_period", "rainfall", "temperature"),
    ).to_csv(csv, index=False)
    build_counterfactual_cmd(
        csv,
        ["rainfall=seasonal_min", "temperature=x-5"],
        start_time_period="2023-06",
    )
    out = pd.read_csv(tmp_path / "data_cf.csv")
    assert out["rainfall"][2] == 40.0
    assert out["temperature"][2] == 5.0
    assert out["temperature"][0] == 10.0  # outside active range, unchanged
