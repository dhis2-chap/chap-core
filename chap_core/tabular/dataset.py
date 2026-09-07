"""Load and validate a fully preprocessed tabular dataset.

The dataset must already be model-ready: numeric or encoded, no missing values,
deduplicated, one target, global analysis only. The assumptions are *checked*,
not assumed - :func:`load_tabular_dataset` refuses to run and names the
offending columns when any of them is violated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
from pandas.api.types import is_numeric_dtype

if TYPE_CHECKING:
    from pathlib import Path


class DatasetAssumptionError(ValueError):
    """Raised when the input dataset violates an input assumption."""


@dataclass(frozen=True)
class TabularDataset:
    """A validated, fully preprocessed tabular dataset.

    ``features`` is a numeric, missing-free design matrix and ``target`` is the
    single target column. There is no location or time structure - this is
    global analysis only.
    """

    features: pd.DataFrame
    target: pd.Series
    target_name: str

    @property
    def feature_names(self) -> list[str]:
        return list(self.features.columns)

    def __len__(self) -> int:
        return len(self.target)


def load_tabular_dataset(csv_path: str | Path, target: str = "target") -> TabularDataset:
    """Read ``csv_path`` and return a validated :class:`TabularDataset`.

    Raises :class:`DatasetAssumptionError`, naming the offending columns, on a
    missing target column, missing values, non-numeric unencoded columns, exact
    duplicate rows, or a constant target.
    """
    frame = pd.read_csv(csv_path)
    _check_assumptions(frame, target)
    return TabularDataset(
        features=frame.drop(columns=[target]),
        target=frame[target],
        target_name=target,
    )


def validate_feature_frame(frame: pd.DataFrame) -> None:
    """Check the shared feature assumptions: no missing values, all numeric.

    Raises :class:`DatasetAssumptionError` naming the offending columns.
    """
    missing = [col for col in frame.columns if frame[col].isna().any()]
    if missing:
        raise DatasetAssumptionError(f"Columns contain missing values: {missing}")

    non_numeric = [col for col in frame.columns if not is_numeric_dtype(frame[col])]
    if non_numeric:
        raise DatasetAssumptionError(f"Non-numeric unencoded columns: {non_numeric}. Encode them first.")


def _check_assumptions(frame: pd.DataFrame, target: str) -> None:
    if target not in frame.columns:
        raise DatasetAssumptionError(f"Target column {target!r} not found. Available columns: {list(frame.columns)}")

    validate_feature_frame(frame)

    duplicate_count = int(frame.duplicated().sum())
    if duplicate_count:
        raise DatasetAssumptionError(f"Dataset has {duplicate_count} exact duplicate row(s).")

    if frame[target].nunique(dropna=False) < 2:
        raise DatasetAssumptionError(f"Target column {target!r} is constant.")
