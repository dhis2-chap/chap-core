import math
from unittest.mock import patch

import pandas as pd
import pytest

from chap_core.cli_endpoints.build_counterfactual import build_counterfactual_cmd
from chap_core.cli_endpoints.build_counterfactual import FeatureTransformations

_LOCATIONS = ["A"]
_PERIODS = ["2022-01", "2022-02", "2022-03"]


# --- validate_expression ---


def testvalidate_expression_addition():
    FeatureTransformations.validate_expression("x+10")


def testvalidate_expression_subtraction():
    FeatureTransformations.validate_expression("x-30")


def testvalidate_expression_multiplication():
    FeatureTransformations.validate_expression("x*0.1")


def testvalidate_expression_division():
    FeatureTransformations.validate_expression("x/2")


def testvalidate_expression_composite():
    FeatureTransformations.validate_expression("x*0+1")


def testvalidate_expression_inverted():
    FeatureTransformations.validate_expression("1-x")


def testvalidate_expression_abs():
    FeatureTransformations.validate_expression("abs(x)")


def testvalidate_expression_round():
    FeatureTransformations.validate_expression("round(x)")


def testvalidate_expression_nested():
    FeatureTransformations.validate_expression("abs(x*0.1-5)")


def testvalidate_expression_disallowed_name():
    with pytest.raises(ValueError, match="Disallowed name"):
        FeatureTransformations.validate_expression("y+1")


def testvalidate_expression_disallowed_function():
    with pytest.raises(ValueError, match="Disallowed function"):
        FeatureTransformations.validate_expression("int(x)")


def testvalidate_expression_string_constant():
    with pytest.raises(ValueError, match="Non-numeric constant"):
        FeatureTransformations.validate_expression("x+'a'")


def testvalidate_expression_syntax_error():
    with pytest.raises(ValueError, match="Invalid expression"):
        FeatureTransformations.validate_expression("x +* 1")


def testvalidate_expression_exponentiation():
    FeatureTransformations.validate_expression("x**2")


def testvalidate_expression_disallowed_operator():
    with pytest.raises(ValueError):
        FeatureTransformations.validate_expression("x // 2")


# --- parse_transformations ---


def testparse_transformations_single():
    assert FeatureTransformations.parse_transformations(["rainfall=x+10"]) == [("rainfall", "x+10")]


def testparse_transformations_multiple():
    result = FeatureTransformations.parse_transformations(["rainfall=x*0.01", "temperature=x-30"])
    assert result == [("rainfall", "x*0.01"), ("temperature", "x-30")]


def testparse_transformations_splits_at_first_equals():
    # expression itself contains =  → only first = is the separator
    result = FeatureTransformations.parse_transformations(["col=x*0+1"])
    assert result == [("col", "x*0+1")]


def testparse_transformations_missing_separator():
    with pytest.raises(ValueError, match="not in 'column=expression' format"):
        FeatureTransformations.parse_transformations(["rainfall"])


# --- apply_transformation ---


def testapply_transformation_basic():
    s = pd.Series([1.0, 2.0, 3.0])
    result = FeatureTransformations.apply_transformation(s, "x+10")
    assert list(result) == [11.0, 12.0, 13.0]


def testapply_transformation_nan_unchanged():
    s = pd.Series([1.0, float("nan"), 3.0])
    result = FeatureTransformations.apply_transformation(s, "x*2")
    assert result[0] == 2.0
    assert math.isnan(result[1])
    assert result[2] == 6.0


def testapply_transformation_abs():
    s = pd.Series([-3.0, 2.0])
    result = FeatureTransformations.apply_transformation(s, "abs(x)")
    assert list(result) == [3.0, 2.0]


def testapply_transformation_round():
    s = pd.Series([1.4, 2.6])
    result = FeatureTransformations.apply_transformation(s, "round(x)")
    assert list(result) == [1, 3]


def testapply_transformation_one_minus_x():
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
