"""Tests for the dataset validation CLI command and service."""

from pathlib import Path

import pandas as pd
import pytest

from chap_core.datatypes import FullData
from chap_core.external.model_configuration import ModelTemplateConfigV2
from chap_core.cli_endpoints.validate import _format_period_ranges, _report_period_gaps
from chap_core.services.dataset_validation import ValidationIssue, check_unused_covariates, validate_dataset
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


def test_validate_warns_on_non_numeric_field(tmp_path):
    raw_df = pd.read_csv(LAOS_SUBSET)
    raw_df["notes"] = "some text"
    csv_path = tmp_path / "with_string_col.csv"
    raw_df.to_csv(csv_path, index=False)
    dataset = DataSet.from_csv(csv_path)
    issues = validate_dataset(dataset)
    warnings = [i for i in issues if i.level == "warning"]
    assert any("notes" in w.message and "non-numeric" in w.message for w in warnings)


def test_validate_accepts_integer_columns(tmp_path):
    raw_df = pd.read_csv(LAOS_SUBSET)
    raw_df["population"] = 100000
    csv_path = tmp_path / "with_int_col.csv"
    raw_df.to_csv(csv_path, index=False)
    dataset = DataSet.from_csv(csv_path)
    issues = validate_dataset(dataset)
    assert not any("population" in i.message for i in issues)


def test_validate_skips_location_name_field(tmp_path):
    raw_df = pd.read_csv(LAOS_SUBSET)
    raw_df["location_name"] = "Some Name"
    csv_path = tmp_path / "with_location_name.csv"
    raw_df.to_csv(csv_path, index=False)
    dataset = DataSet.from_csv(csv_path)
    issues = validate_dataset(dataset)
    assert not any("location_name" in i.message for i in issues)


def test_validate_warns_on_unused_covariate():
    dataset = DataSet.from_csv(LAOS_SUBSET, FullData)
    config = ModelTemplateConfigV2(name="test_model", required_covariates=["rainfall"])
    issues = validate_dataset(dataset, model_template_config=config)
    warnings = [i for i in issues if i.level == "warning"]
    assert any("mean_temperature" in w.message and "not used by the model" in w.message for w in warnings)


def test_validate_warns_unused_when_allow_free_but_no_additional():
    dataset = DataSet.from_csv(LAOS_SUBSET, FullData)
    config = ModelTemplateConfigV2(
        name="test_model",
        required_covariates=[],
        allow_free_additional_continuous_covariates=True,
    )
    issues = validate_dataset(dataset, model_template_config=config)
    warnings = [i for i in issues if i.level == "warning"]
    assert any("not used by the model" in w.message for w in warnings)


def test_validate_no_unused_warning_for_configured_additional_covariates():
    dataset = DataSet.from_csv(LAOS_SUBSET, FullData)
    config = ModelTemplateConfigV2(
        name="test_model",
        required_covariates=["rainfall"],
        allow_free_additional_continuous_covariates=True,
    )
    issues = validate_dataset(
        dataset,
        model_template_config=config,
        additional_continuous_covariates=["mean_temperature", "population"],
    )
    warnings = [i for i in issues if i.level == "warning"]
    assert not any("not used by the model" in w.message for w in warnings)


def test_check_unused_covariates_flags_extra_column():
    dataset = DataSet.from_csv(LAOS_SUBSET, FullData)
    config = ModelTemplateConfigV2(name="test_model", required_covariates=["rainfall"])
    issues = check_unused_covariates(dataset, config)
    assert all(i.level == "warning" for i in issues)
    messages = [i.message for i in issues]
    assert any("mean_temperature" in m and "not used by the model" in m for m in messages)


def test_check_unused_covariates_no_issues_when_all_used():
    dataset = DataSet.from_csv(LAOS_SUBSET, FullData)
    # laos_subset has covariates: rainfall, mean_temperature, population
    config = ModelTemplateConfigV2(
        name="test_model", required_covariates=["rainfall", "mean_temperature", "population"]
    )
    issues = check_unused_covariates(dataset, config)
    assert issues == []


def test_check_unused_covariates_respects_additional_continuous_covariates():
    dataset = DataSet.from_csv(LAOS_SUBSET, FullData)
    config = ModelTemplateConfigV2(
        name="test_model",
        required_covariates=["rainfall"],
        allow_free_additional_continuous_covariates=True,
    )
    issues = check_unused_covariates(
        dataset, config, additional_continuous_covariates=["mean_temperature", "population"]
    )
    assert issues == []


def test_check_unused_covariates_warns_when_allow_free_but_column_not_listed():
    dataset = DataSet.from_csv(LAOS_SUBSET, FullData)
    config = ModelTemplateConfigV2(
        name="test_model",
        required_covariates=["rainfall"],
        allow_free_additional_continuous_covariates=True,
    )
    issues = check_unused_covariates(dataset, config, additional_continuous_covariates=[])
    assert any("mean_temperature" in i.message for i in issues)


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
