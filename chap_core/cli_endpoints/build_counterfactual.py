"""Build counterfactual CSV by applying column transformations."""

from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003 — used at runtime via cyclopts get_type_hints()
from typing import Annotated

import pandas as pd
from cyclopts import Parameter

from chap_core.cli_endpoints.expressions import apply_transformation, parse_transformations, validate_expression

logger = logging.getLogger(__name__)


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

    pairs = parse_transformations(transformations)

    df = pd.read_csv(dataset_csv)

    invalid_cols = [c for c in df.columns if "=" in c]
    if invalid_cols:
        raise ValueError(f"Column names must not contain '=': {invalid_cols}")

    for _, expr in pairs:
        validate_expression(expr)

    for col, _ in pairs:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset")

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
            df[col] = apply_transformation(df[col], expr)
        else:
            df.loc[row_mask, col] = apply_transformation(df.loc[row_mask, col], expr)
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
