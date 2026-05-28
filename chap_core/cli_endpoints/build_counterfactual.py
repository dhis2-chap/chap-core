"""Build counterfactual CSV by applying column transformations."""

from __future__ import annotations

import ast
import logging
from pathlib import Path  # noqa: TC003 — used at runtime via cyclopts get_type_hints()
from typing import Annotated

import pandas as pd
from cyclopts import Parameter

logger = logging.getLogger(__name__)


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


def build_counterfactual_cmd(
    dataset_csv: Annotated[Path, Parameter(help="Path to input CSV file")],
    transformations: Annotated[
        list[str],
        Parameter(help="Column transformations as 'column=expression' pairs (e.g. rainfall=x*0.01)"),
    ],
    *,
    start_time_period: Annotated[
        str | None,
        Parameter("--start-time-period", help="Apply transformation from this period onward (inclusive)"),
    ] = None,
    end_time_period: Annotated[
        str | None,
        Parameter("--end-time-period", help="Apply transformation up to this period (inclusive)"),
    ] = None,
    output_csv: Annotated[
        Path | None,
        Parameter("--output-csv", help="Output CSV path; defaults to input filename with '_cf' suffix"),
    ] = None,
) -> None:
    """Build a counterfactual CSV by applying arithmetic transformations to selected columns.

    Reads dataset_csv, applies each 'column=expression' transformation to non-missing values
    in the specified column, optionally restricted to a time-period range, and writes the
    result to output_csv.

    Expression grammar
    ------------------
    Each transformation is written as ``column=expression``, where ``expression`` is an
    arithmetic formula over the placeholder ``x`` (the original column value).

    Allowed:
      - Numeric literals: integers and floats (e.g. 10, 0.5)
      - Variable: ``x`` (the current cell value)
      - Operators: ``+``, ``-``, ``*``, ``/``, ``**``
      - Unary minus/plus: ``-x``, ``+x``
      - Functions: ``abs(x)``, ``round(x)``
      - Nesting and composition: ``abs(x*0.1-5)``, ``round(x+0.5)``

    Not allowed:
      - Any name other than ``x`` (e.g. ``y``, ``pi``)
      - Any operator not listed above
      - String literals or boolean constants
      - Any function call other than ``abs`` and ``round``
      - Column names containing ``=``

    Missing values (NaN) are left unchanged regardless of the expression.

    Examples:
        chap causal build-counterfactual data.csv rainfall=x*0.01 temperature=x-30
        chap causal build-counterfactual data.csv rainfall=abs(x) --start-time-period 2022-06
        chap causal build-counterfactual data.csv cases=round(x*1.1) --output-csv data_cf.csv
    """
    from chap_core.time_period import TimePeriod

    pairs = FeatureTransformations.parse_transformations(transformations)

    df = pd.read_csv(dataset_csv)

    invalid_cols = [c for c in df.columns if "=" in c]
    if invalid_cols:
        raise ValueError(f"Column names must not contain '=': {invalid_cols}")

    for col, _ in pairs:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset")

    for _, expr in pairs:
        FeatureTransformations.validate_expression(expr)

    if (start_time_period or end_time_period) and "time_period" not in df.columns:
        raise ValueError("Column 'time_period' not found in dataset")

    row_mask = None
    if start_time_period or end_time_period:
        parsed = df["time_period"].apply(TimePeriod.parse)
        row_mask = pd.Series([True] * len(df), index=df.index)
        if start_time_period:
            row_mask &= parsed >= TimePeriod.parse(start_time_period)
        if end_time_period:
            row_mask &= parsed <= TimePeriod.parse(end_time_period)

    for col, expr in pairs:
        original_dtype = df[col].dtype
        if row_mask is None:
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

    out_path = output_csv or dataset_csv.with_stem(dataset_csv.stem + "_cf")
    df.to_csv(out_path, index=False)
    logger.info("Counterfactual CSV written to %s", out_path)
