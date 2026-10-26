"""Masking `disease_cases` in a dataset's masked ranges for a run (CLIM-1300).

A dataset with masked ranges is masked when it is saved: `disease_cases` inside the ranges is
stored as NaN and the original values are not kept. Each run adds `disease_cases_masked` to its
in-memory copy when it loads the data: 1 inside a range and 0 elsewhere, so a model can tell a
masked period from a genuinely missing one (NaN with indicator 0). The CLI masks only the loaded copy.

Future data of a masked run carries the indicator as all 0: a forecast period is never masked,
and models that stack history and future, or fit on every column, need the same columns in both.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

from chap_core.database.dataset_tables import (
    MaskedRange,
    check_masked_ranges_match_dataset,
    parse_masked_period,
)

if TYPE_CHECKING:
    from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
    from chap_core.time_period import PeriodRange, TimePeriod

TARGET_FIELD = "disease_cases"
MASK_INDICATOR_FIELD = "disease_cases_masked"


def apply_target_mask(dataset: DataSet, masked_ranges: list[MaskedRange] | None) -> DataSet:
    """Return a copy of `dataset` with `disease_cases` masked inside the ranges and the indicator added.

    Without ranges the dataset is returned unchanged, with no indicator, so unmasked datasets reach
    models exactly as before. The indicator is computed from the ranges alone, so masking twice gives
    the same result. Run it after any padding, or padded periods get a NaN indicator.
    """
    # Imported here: the weather providers import this module while chap_core.datatypes is still loading.
    from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

    if not masked_ranges:
        return dataset
    if TARGET_FIELD not in dataset.field_names():
        raise ValueError(f"Masked ranges mask {TARGET_FIELD!r}, but the dataset has no {TARGET_FIELD!r} column")
    check_masked_ranges_match_dataset(masked_ranges, dataset.period_range[0])
    bounds = [(parse_masked_period(r.start), parse_masked_period(r.end)) for r in masked_ranges]
    names = [name for name in dataset.field_names() if name != MASK_INDICATOR_FIELD]
    data_class = _with_indicator_class(next(iter(dataset.values())))
    masked = {}
    for location, data in dataset.items():
        inside = _inside_ranges(data.time_period, bounds)
        values = {name: getattr(data, name) for name in names}
        values[TARGET_FIELD] = np.where(inside, np.nan, values[TARGET_FIELD].astype(float))
        values[MASK_INDICATOR_FIELD] = inside.astype(float)
        masked[location] = data_class(data.time_period, **values)
    return DataSet(masked, polygons=dataset.polygons, metadata=dataset.metadata)


def _inside_ranges(period_range: PeriodRange, bounds: list[tuple[TimePeriod, TimePeriod]]) -> np.ndarray:
    """Mark the periods inside any of the inclusive bounds, using index lookups instead of comparing each period."""
    inside = np.zeros(len(period_range), dtype=bool)
    for start, end in bounds:
        inside[period_range.searchsorted(start) : period_range.searchsorted(end, side="right")] = True
    return inside


def check_mask_outside_evaluation_window(dataset: DataSet, n_periods: int, n_splits: int, stride: int) -> None:
    """Raise if a masked period falls in the evaluation window of a backtest on `dataset`.

    The window is the last `n_periods + (n_splits - 1) * stride` periods, which covers every split's
    test truth and the history used inside the window, whatever `n_retrain` is.
    """
    if MASK_INDICATOR_FIELD not in dataset.field_names():
        return
    evaluation_length = n_periods + (n_splits - 1) * stride
    window = dataset.period_range[-evaluation_length:]
    for data in dataset.values():
        masked = np.flatnonzero(getattr(data, MASK_INDICATOR_FIELD)[-evaluation_length:] == 1)
        if len(masked):
            raise ValueError(
                f"Masked period {window[int(masked[0])].id} falls in the evaluation window, which starts at "
                f"{window[0].id}. Backtests cannot evaluate on masked data; use fewer periods or splits, "
                "or a dataset whose masked ranges end before the window."
            )


def check_no_mask_indicator_input(feature_names: list[str] | list[str | None]) -> None:
    """Raise if input data carries the indicator column, which masking adds itself."""
    if MASK_INDICATOR_FIELD in feature_names:
        raise ValueError(f"{MASK_INDICATOR_FIELD!r} is reserved: it is added by masking. Send maskedRanges instead.")


def check_target_column_allows_masking(masked_ranges: list[MaskedRange] | None, target_column: str) -> None:
    """Raise if a dataset with masked ranges is run with a target other than `disease_cases`.

    Ranges mask `disease_cases` only. Masking before the alternate target replaces it would leave the
    indicator beside an unmasked target, and masking after would mask a different variable.
    """
    if masked_ranges and target_column != TARGET_FIELD:
        raise ValueError(
            f"Target column {target_column!r} cannot be used on a dataset with masked ranges; "
            f"masked ranges apply to {TARGET_FIELD!r} only"
        )


def _with_indicator_class(data) -> type:
    """The data's own dataclass plus a float mask indicator, keeping every other column's type."""
    from chap_core.datatypes import TimeSeriesData, tsdataclass

    fields = [
        (field.name, float if field.name == TARGET_FIELD else field.type)
        for field in dataclasses.fields(data)
        if field.name != MASK_INDICATOR_FIELD
    ]
    return tsdataclass(  # type: ignore[no-any-return]
        dataclasses.make_dataclass(
            type(data).__name__, [*fields, (MASK_INDICATOR_FIELD, float)], bases=(TimeSeriesData,)
        )
    )


def with_unmasked_indicator(dataset: DataSet) -> DataSet:
    """Return a copy of `dataset` whose mask indicator is 0 everywhere, adding the column if it is missing."""
    from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

    names = [name for name in dataset.field_names() if name != MASK_INDICATOR_FIELD]
    data_class = _with_indicator_class(next(iter(dataset.values())))
    unmasked = {
        location: data_class(
            data.time_period,
            **{name: getattr(data, name) for name in names},
            **{MASK_INDICATOR_FIELD: np.zeros(len(data.time_period))},
        )
        for location, data in dataset.items()
    }
    return DataSet(unmasked, polygons=dataset.polygons, metadata=dataset.metadata)


def as_future_data(dataset: DataSet, target: str = TARGET_FIELD) -> DataSet:
    """Turn observed data into future data for a model: drop the target, and set the mask indicator to 0."""
    future: DataSet = dataset.remove_field(target)
    if MASK_INDICATOR_FIELD in future.field_names():
        return with_unmasked_indicator(future)
    return future


def without_mask_indicator(dataset: DataSet) -> DataSet:
    """Remove the mask indicator if the dataset has one."""
    if MASK_INDICATOR_FIELD in dataset.field_names():
        stripped: DataSet = dataset.remove_field(MASK_INDICATOR_FIELD)
        return stripped
    return dataset


def without_target(dataset: DataSet, target: str = TARGET_FIELD) -> DataSet:
    """Remove the target and the mask indicator, leaving only covariates (for fitting a weather model)."""
    stripped: DataSet = without_mask_indicator(dataset).remove_field(target)
    return stripped
