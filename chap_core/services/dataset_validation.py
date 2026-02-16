"""Validation logic for CHAP datasets."""

import logging

import numpy as np
import pandas as pd
from pydantic import BaseModel

from chap_core.external.model_configuration import ModelTemplateConfigV2
from chap_core.model_spec import PeriodType
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import Month, Week

logger = logging.getLogger(__name__)


class ValidationIssue(BaseModel):
    level: str  # "error" or "warning"
    message: str
    location: str | None = None
    time_periods: list[str] | None = None


def validate_dataset(
    dataset: DataSet,
    raw_df: pd.DataFrame | None = None,
    model_template_config: ModelTemplateConfigV2 | None = None,
) -> list[ValidationIssue]:
    """Validate a dataset for CHAP compatibility.

    Parameters
    ----------
    dataset : DataSet
        The loaded dataset to validate.
    raw_df : pd.DataFrame | None
        The raw DataFrame before loading (used for location completeness check).
    model_template_config : ModelTemplateConfigV2 | None
        Optional model config to validate against.

    Returns
    -------
    list[ValidationIssue]
        List of validation issues found.
    """
    issues: list[ValidationIssue] = []

    issues.extend(_check_nan_covariates(dataset))

    if raw_df is not None:
        issues.extend(_check_location_completeness(raw_df))

    if model_template_config is not None:
        issues.extend(_check_required_covariates(dataset, model_template_config))
        issues.extend(_check_period_type(dataset, model_template_config))

    return issues


RESERVED_FIELDS = {"time_period", "location", "location_name", "disease_cases"}


def _check_nan_covariates(dataset: DataSet) -> list[ValidationIssue]:
    """Check for NaN values in covariate columns (excluding disease_cases)."""
    issues: list[ValidationIssue] = []
    field_names = dataset.field_names()
    non_numeric_warned: set[str] = set()
    for location, data in dataset.items():
        for field_name in field_names:
            if field_name in RESERVED_FIELDS:
                continue
            values = getattr(data, field_name)
            if not np.issubdtype(values.dtype, np.number):
                if field_name not in non_numeric_warned:
                    non_numeric_warned.add(field_name)
                    issues.append(
                        ValidationIssue(
                            level="warning",
                            message=f"Column '{field_name}' is non-numeric and will be ignored by models",
                        )
                    )
                continue
            if not np.issubdtype(values.dtype, np.floating):
                continue
            isnan = np.isnan(values)
            if np.any(isnan):
                nan_periods = [data.time_period[i].to_string() for i in np.flatnonzero(isnan)]
                issues.append(
                    ValidationIssue(
                        level="error",
                        message=f"Missing values in '{field_name}'",
                        location=location,
                        time_periods=nan_periods,
                    )
                )
    return issues


def _check_location_completeness(raw_df: pd.DataFrame) -> list[ValidationIssue]:
    """Check that every location has the same set of time periods."""
    issues: list[ValidationIssue] = []
    periods_per_location = raw_df.groupby("location")["time_period"].apply(set)
    all_periods = set().union(*periods_per_location)
    for location, periods in periods_per_location.items():
        missing = all_periods - periods
        if missing:
            issues.append(
                ValidationIssue(
                    level="error",
                    message="Location has fewer time periods than others",
                    location=str(location),
                    time_periods=sorted(missing),
                )
            )
    return issues


def _check_required_covariates(dataset: DataSet, config: ModelTemplateConfigV2) -> list[ValidationIssue]:
    """Check that all required covariates from the model config are present."""
    dataset_fields = set(dataset.field_names())
    return [
        ValidationIssue(
            level="error",
            message=f"Required covariate '{covariate}' not found in dataset. "
            f"Available fields: {sorted(dataset_fields)}",
        )
        for covariate in config.required_covariates
        if covariate not in dataset_fields
    ]


def _check_period_type(dataset: DataSet, config: ModelTemplateConfigV2) -> list[ValidationIssue]:
    """Check that the dataset period type matches the model's supported period type."""
    issues: list[ValidationIssue] = []
    if config.supported_period_type == PeriodType.any:
        return issues

    first_period = dataset.period_range[0]
    if config.supported_period_type == PeriodType.month and not isinstance(first_period, Month):
        issues.append(
            ValidationIssue(
                level="error",
                message=f"Model requires monthly data but dataset has {type(first_period).__name__.lower()} periods",
            )
        )
    elif config.supported_period_type == PeriodType.week and not isinstance(first_period, Week):
        issues.append(
            ValidationIssue(
                level="error",
                message=f"Model requires weekly data but dataset has {type(first_period).__name__.lower()} periods",
            )
        )
    return issues
