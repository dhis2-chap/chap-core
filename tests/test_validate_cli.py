"""Tests for the dataset validation CLI command and service."""

from pathlib import Path

import pandas as pd
import pytest

from chap_core.datatypes import FullData
from chap_core.external.model_configuration import ModelTemplateConfigV2
from chap_core.cli_endpoints.validate import _format_period_ranges, _report_period_gaps
from chap_core.services.dataset_validation import ValidationIssue, validate_dataset
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

from chap_core.model_spec import PeriodType

EXAMPLE_DATA = Path("example_data")
LAOS_SUBSET = EXAMPLE_DATA / "laos_subset.csv"
NICARAGUA_WEEKLY_SUBSET = EXAMPLE_DATA / "nicaragua_weekly_subset.csv"
FAULTY_DATA = EXAMPLE_DATA / "faulty_datasets"


def test_validate_valid_dataset():
    dataset = DataSet.from_csv(LAOS_SUBSET, FullData)
    raw_df = pd.read_csv(LAOS_SUBSET)
    issues = validate_dataset(dataset, raw_df=raw_df)
    errors = [i for i in issues if i.level == "error"]
    assert len(errors) == 0


def test_validate_missing_covariate_values():
    csv_path = FAULTY_DATA / "missing_covariate_values.csv"
    dataset = DataSet.from_csv(csv_path, FullData)
    issues = validate_dataset(dataset)
    errors = [i for i in issues if i.level == "error"]
    assert len(errors) > 0
    assert any("rainfall" in e.message for e in errors)
    assert any(e.location == "Bokeo" for e in errors)


def test_validate_non_consecutive_periods():
    csv_path = FAULTY_DATA / "non_consecutive_periods.csv"
    with pytest.raises(ValueError, match="consecutive"):
        DataSet.from_csv(csv_path, FullData)


def test_report_period_gaps(capsys):
    csv_path = FAULTY_DATA / "non_consecutive_periods.csv"
    raw_df = pd.read_csv(csv_path)
    _report_period_gaps(raw_df)
    output = capsys.readouterr().out
    assert "missing" in output


def test_format_period_ranges_groups_consecutive():
    assert _format_period_ranges(["2008-01"]) == "2008-01"
    result = _format_period_ranges(["2008-01", "2008-02", "2008-03"])
    assert "2008-01 to 2008-03 (3 periods)" in result


def test_format_period_ranges_splits_non_consecutive():
    result = _format_period_ranges(["2008-01", "2008-02", "2008-06"])
    assert "2008-01 to 2008-02 (2 periods)" in result
    assert "2008-06" in result


def test_validate_incomplete_locations():
    csv_path = FAULTY_DATA / "incomplete_locations.csv"
    raw_df = pd.read_csv(csv_path)
    dataset = DataSet.from_csv(csv_path, FullData)
    issues = validate_dataset(dataset, raw_df=raw_df)
    errors = [i for i in issues if i.level == "error"]
    assert len(errors) > 0
    location_issues = [e for e in errors if e.location == "Bokeo"]
    assert len(location_issues) > 0


def test_validate_with_model_missing_covariate():
    dataset = DataSet.from_csv(LAOS_SUBSET, FullData)
    config = ModelTemplateConfigV2(
        name="test_model",
        required_covariates=["rainfall", "humidity"],
    )
    issues = validate_dataset(dataset, model_template_config=config)
    errors = [i for i in issues if i.level == "error"]
    assert len(errors) > 0
    assert any("humidity" in e.message for e in errors)


def test_validate_valid_weekly_dataset():
    dataset = DataSet.from_csv(NICARAGUA_WEEKLY_SUBSET, FullData)
    raw_df = pd.read_csv(NICARAGUA_WEEKLY_SUBSET)
    issues = validate_dataset(dataset, raw_df=raw_df)
    errors = [i for i in issues if i.level == "error"]
    assert len(errors) == 0


def test_validate_missing_covariate_values_weekly():
    csv_path = FAULTY_DATA / "missing_covariate_values_weekly.csv"
    dataset = DataSet.from_csv(csv_path, FullData)
    issues = validate_dataset(dataset)
    errors = [i for i in issues if i.level == "error"]
    assert len(errors) > 0
    assert any("mean_temperature" in e.message for e in errors)
    assert any(e.location == "boaco" for e in errors)


def test_validate_non_consecutive_periods_weekly():
    csv_path = FAULTY_DATA / "non_consecutive_periods_weekly.csv"
    with pytest.raises(ValueError, match="consecutive"):
        DataSet.from_csv(csv_path, FullData)


def test_validate_incomplete_locations_weekly():
    csv_path = FAULTY_DATA / "incomplete_locations_weekly.csv"
    raw_df = pd.read_csv(csv_path)
    dataset = DataSet.from_csv(csv_path, FullData)
    issues = validate_dataset(dataset, raw_df=raw_df)
    errors = [i for i in issues if i.level == "error"]
    assert len(errors) > 0
    location_issues = [e for e in errors if e.location == "boaco"]
    assert len(location_issues) > 0


def test_validate_weekly_data_against_monthly_model():
    dataset = DataSet.from_csv(NICARAGUA_WEEKLY_SUBSET, FullData)
    config = ModelTemplateConfigV2(
        name="test_monthly_model",
        supported_period_type=PeriodType.month,
        required_covariates=["rainfall"],
    )
    issues = validate_dataset(dataset, model_template_config=config)
    errors = [i for i in issues if i.level == "error"]
    assert len(errors) > 0
    assert any("monthly" in e.message for e in errors)
