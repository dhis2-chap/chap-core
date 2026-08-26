"""Build counterfactual CSV by applying column transformations."""

from __future__ import annotations

import ast
import logging
from pathlib import Path  # noqa: TC003 — used at runtime via cyclopts get_type_hints()
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from cyclopts import Parameter

from chap_core.cli_endpoints._common import resolve_period_arg

logger = logging.getLogger(__name__)

_SEASONAL_KEYWORDS = frozenset({"seasonal_min", "seasonal_max", "window_avg_min", "window_avg_max"})


class FeatureTransformations:
    """Safe arithmetic expression parsing, validation, and evaluation for counterfactual transforms."""

    def __new__(cls, *args, **kwargs):
        raise TypeError(f"{cls.__name__} instances cannot be created")

    _ALLOWED_FUNCS = frozenset({"abs", "round"})

    _ALLOWED_NODE_TYPES = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Name,
        ast.Constant,
        ast.Call,
        ast.Load,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
    )

    @classmethod
    def validate_expression(cls, expr: str) -> None:
        """Raise ValueError if expr is not a safe arithmetic expression in x."""
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid expression '{expr}': {e}") from e

        for node in ast.walk(tree):
            if not isinstance(node, FeatureTransformations._ALLOWED_NODE_TYPES):
                raise ValueError(f"Disallowed construct in expression '{expr}'")
            if isinstance(node, ast.Name) and node.id not in FeatureTransformations._ALLOWED_FUNCS and node.id != "x":
                raise ValueError(f"Disallowed name '{node.id}' in expression '{expr}'")
            if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
                raise ValueError(f"Non-numeric constant in expression '{expr}'")
            if isinstance(node, ast.Call) and not (
                isinstance(node.func, ast.Name) and node.func.id in FeatureTransformations._ALLOWED_FUNCS
            ):
                raise ValueError(f"Disallowed function call in expression '{expr}'")

    @classmethod
    def parse_transformations(cls, transformations: list[str]) -> list[tuple[str, str]]:
        """Parse ['col=expr', ...] into [('col', 'expr'), ...].

        Raises ValueError if any entry lacks an '=' separator.
        """
        result = []
        for t in transformations:
            if "=" not in t:
                raise ValueError(f"Transformation '{t}' is not in 'column=expression' format")
            col, expr = t.split("=", 1)
            result.append((col, expr))
        return result

    @classmethod
    def apply_transformation(cls, series: pd.Series, expr: str) -> pd.Series:
        """Apply expr to each non-NaN value; leave NaN values unchanged."""

        cls.validate_expression(expr)

        namespace = {"__builtins__": {}, "abs": abs, "round": round}
        result = series.copy()
        mask = ~series.isna()
        code = compile(expr, "<expr>", "eval")
        result[mask] = series[mask].apply(lambda x: eval(code, namespace, {"x": x}))
        return result


class SeasonalAggregation:
    """Aggregation-based counterfactual values, computed per location from rows outside the active range.

    "Historical" here means every row not selected by the active (--start-time-period/--end-time-period) mask,
    which is chronologically earlier only in the common case where the active range runs to the end of the
    dataset. If --end-time-period leaves later rows out of the active range, those later rows count as
    "historical" too.

    Unlike FeatureTransformations, these always overwrite the target cell, even if it is currently NaN — the
    point is to fill in unset forecast-horizon covariate values from the location's own history.
    """

    def __new__(cls, *args, **kwargs):
        raise TypeError(f"{cls.__name__} instances cannot be created")

    @staticmethod
    def period_of_year_key(period) -> int:
        """Return the within-year key (month number or ISO week number) used to match same-season periods."""
        from chap_core.time_period import Month, Week

        if isinstance(period, Month):
            return int(period.month)
        if isinstance(period, Week):
            return int(period.week)
        raise ValueError(f"seasonal_min/seasonal_max require Month or Week time periods, got {type(period).__name__}")

    @classmethod
    def seasonal_extremum(
        cls,
        df: pd.DataFrame,
        col: str,
        active_mask: pd.Series,
        parsed_periods: pd.Series,
        agg: Literal["min", "max"],
    ) -> pd.Series:
        """For each active row, use the min/max of `col` across rows outside the active range that share the
        same location and period-of-year (e.g. same calendar month, any other year)."""
        keys = parsed_periods.apply(cls.period_of_year_key)
        historical = pd.DataFrame({"location": df["location"], "key": keys, "value": df[col]})[~active_mask]
        grouped = historical.groupby(["location", "key"])["value"].agg(agg)

        result = df[col].copy()
        for idx in df.index[active_mask]:
            group_key = (df.at[idx, "location"], keys.at[idx])
            value = grouped.get(group_key)
            if value is None or pd.isna(value):
                logger.warning(
                    "No historical data for location '%s', column '%s', period-of-year %s; value left unchanged",
                    group_key[0],
                    col,
                    group_key[1],
                )
                continue
            result.at[idx] = value
        return result

    @classmethod
    def window_average_extremum(
        cls,
        df: pd.DataFrame,
        col: str,
        active_mask: pd.Series,
        parsed_periods: pd.Series,
        agg: Literal["min", "max"],
    ) -> pd.Series:
        """For each location, average the min/max of `col` over consecutive, non-overlapping windows of rows
        outside the active range, each the same length as that location's active range, then assign the
        scalar to every active row."""
        result = df[col].copy()
        for location, location_index in df.groupby("location").groups.items():
            active_index = [i for i in location_index if active_mask.at[i]]
            window_length = len(active_index)
            if window_length == 0:
                continue
            historical_index = sorted(
                (i for i in location_index if not active_mask.at[i]),
                key=lambda i: parsed_periods.at[i],  # type: ignore[arg-type,return-value]
            )

            window_extrema = []
            for start in range(0, len(historical_index), window_length):
                window_index = historical_index[start : start + window_length]
                if len(window_index) < window_length:
                    break
                window_value = df.loc[window_index, col].agg(agg)
                if pd.notna(window_value):
                    window_extrema.append(window_value)

            if not window_extrema:
                logger.warning(
                    "No usable historical window for location '%s', column '%s'; values left unchanged",
                    location,
                    col,
                )
                continue
            result.loc[active_index] = np.mean(window_extrema)
        return result


def _validate_inputs(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    seasonal_pairs: list[tuple[str, str]],
    start_time_period: str | None,
    end_time_period: str | None,
) -> None:
    invalid_cols = [c for c in df.columns if "=" in c]
    if invalid_cols:
        raise ValueError(f"Column names must not contain '=': {invalid_cols}")

    for col, _ in pairs:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset")

    for _, expr in pairs:
        if expr.strip() not in _SEASONAL_KEYWORDS:
            FeatureTransformations.validate_expression(expr)

    if seasonal_pairs:
        if "location" not in df.columns:
            raise ValueError("Column 'location' not found in dataset")
        if start_time_period is None:
            raise ValueError(
                "--start-time-period is required when using seasonal_min/seasonal_max/window_avg_min/window_avg_max"
            )

    if (start_time_period or end_time_period) and "time_period" not in df.columns:
        raise ValueError("Column 'time_period' not found in dataset")


def _build_row_mask(
    df: pd.DataFrame, start_time_period: str | None, end_time_period: str | None
) -> tuple[pd.Series | None, pd.Series | None]:
    """Return (row_mask, parsed_periods), or (None, None) if no time-period bound was given."""
    if not (start_time_period or end_time_period):
        return None, None

    from chap_core.time_period import TimePeriod

    parsed = df["time_period"].apply(TimePeriod.parse)
    row_mask = pd.Series([True] * len(df), index=df.index)
    if start_time_period:
        start_obj = resolve_period_arg(start_time_period, parsed, "start_time_period")
        row_mask &= parsed >= start_obj
    if end_time_period:
        end_obj = resolve_period_arg(end_time_period, parsed, "end_time_period")
        row_mask &= parsed <= end_obj
    return row_mask, parsed


def _apply_column_transformation(
    df: pd.DataFrame,
    col: str,
    expr: str,
    row_mask: pd.Series | None,
    parsed: pd.Series | None,
) -> None:
    """Apply a single 'column=expression' transformation to df in place, warning on dtype changes."""
    original_dtype = df[col].dtype
    keyword = expr.strip()
    if keyword in _SEASONAL_KEYWORDS:
        assert row_mask is not None and parsed is not None  # guaranteed by the seasonal_pairs check in _validate_inputs
        agg: Literal["min", "max"] = "min" if keyword.endswith("min") else "max"
        if keyword.startswith("seasonal"):
            df[col] = SeasonalAggregation.seasonal_extremum(df, col, row_mask, parsed, agg)
        else:
            df[col] = SeasonalAggregation.window_average_extremum(df, col, row_mask, parsed, agg)
    elif row_mask is None:
        df[col] = FeatureTransformations.apply_transformation(df[col], expr)
    else:
        df.loc[row_mask, col] = FeatureTransformations.apply_transformation(df.loc[row_mask, col], expr)

    if df[col].dtype != original_dtype:
        logger.warning(
            "Column '%s' changed type from %s to %s after transformation",
            col,
            original_dtype,
            df[col].dtype,
        )


def build_counterfactual_cmd(
    dataset_csv: Annotated[Path, Parameter(help="Path to input CSV file")],
    transformations: Annotated[
        list[str],
        Parameter(help="Column transformations as 'column=expression' pairs (e.g. rainfall=x*0.01)"),
    ],
    *,
    start_time_period: Annotated[
        str | None,
        Parameter(
            "--start-time-period",
            help=(
                "Apply transformation from this period onward (inclusive). Accepts an exact period "
                "(e.g. '2023-01') or a relative index: '+n' for the n-th period from the start, "
                "'-n' for the n-th period from the end (both 1-based)."
            ),
        ),
    ] = None,
    end_time_period: Annotated[
        str | None,
        Parameter(
            "--end-time-period",
            help=(
                "Apply transformation up to this period (inclusive). Accepts an exact period "
                "(e.g. '2023-01') or a relative index: '+n' for the n-th period from the start, "
                "'-n' for the n-th period from the end (both 1-based)."
            ),
        ),
    ] = None,
    output_csv: Annotated[
        Path | None,
        Parameter("--output-csv", help="Output CSV path; defaults to input filename with '_cf' suffix"),
    ] = None,
) -> None:
    """Build a counterfactual CSV by applying transformations to selected columns.

    Reads dataset_csv, applies each 'column=expression' transformation to the specified column,
    optionally restricted to a time-period range, and writes the result to output_csv.

    Expression grammar
    ------------------
    Each transformation is written as ``column=expression``, where ``expression`` is either an
    arithmetic formula over the placeholder ``x`` (the original column value), or one of four
    reserved seasonal/window keywords.

    Arithmetic — allowed:
      - Numeric literals: integers and floats (e.g. 10, 0.5)
      - Variable: ``x`` (the current cell value)
      - Operators: ``+``, ``-``, ``*``, ``/``, ``**``
      - Unary minus/plus: ``-x``, ``+x``
      - Functions: ``abs(x)``, ``round(x)``
      - Nesting and composition: ``abs(x*0.1-5)``, ``round(x+0.5)``

    Arithmetic — not allowed:
      - Any name other than ``x`` (e.g. ``y``, ``pi``)
      - Any operator not listed above
      - String literals or boolean constants
      - Any function call other than ``abs`` and ``round``
      - Column names containing ``=``

    Missing values (NaN) are left unchanged by arithmetic expressions.

    Seasonal/window keywords (require --start-time-period and a 'location' column). Both are computed
    from rows outside the active (--start-time-period to --end-time-period) range — chronologically
    earlier only in the common case where the active range runs to the end of the dataset; if
    --end-time-period leaves later rows out of the active range, those later rows are pooled in too:
      - ``seasonal_min`` / ``seasonal_max``: per location, the min/max of the column across rows
        sharing the same month-of-year (Month periods) or ISO week-of-year (Week periods).
      - ``window_avg_min`` / ``window_avg_max``: per location, the average of the min/max computed
        over consecutive, non-overlapping windows the same length as the active range.

    Unlike arithmetic expressions, these keywords always overwrite the target cell, even if it is
    currently NaN — they are meant to fill in unset forecast-horizon covariate values. If a
    location has no usable history, its cells are left unchanged and a warning is logged.

    --start-time-period/--end-time-period also accept a relative period index instead of an exact
    period string: '+n' selects the n-th period from the start of the dataset, '-n' selects the
    n-th period from the end (both 1-based, e.g. '-1' is the dataset's last period).

    Examples:
        chap causal build-counterfactual data.csv rainfall=x*0.01 temperature=x-30
        chap causal build-counterfactual data.csv rainfall=abs(x) --start-time-period 2022-06
        chap causal build-counterfactual data.csv cases=round(x*1.1) --output-csv data_cf.csv
        chap causal build-counterfactual data.csv rainfall=seasonal_min --start-time-period 2023-06
        chap causal build-counterfactual data.csv rainfall=window_avg_max --start-time-period 2023-06
        chap causal build-counterfactual data.csv rainfall=x*0.01 --start-time-period=-3
    """
    pairs = FeatureTransformations.parse_transformations(transformations)
    seasonal_pairs = [(col, expr) for col, expr in pairs if expr.strip() in _SEASONAL_KEYWORDS]

    df = pd.read_csv(dataset_csv)
    _validate_inputs(df, pairs, seasonal_pairs, start_time_period, end_time_period)
    row_mask, parsed = _build_row_mask(df, start_time_period, end_time_period)

    for col, expr in pairs:
        _apply_column_transformation(df, col, expr, row_mask, parsed)

    out_path = output_csv or dataset_csv.with_stem(dataset_csv.stem + "_cf")
    df.to_csv(out_path, index=False)
    logger.info("Counterfactual CSV written to %s", out_path)
