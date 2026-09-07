"""Tests for tabular dataset loading and assumption checks."""

import pytest

from chap_core.tabular.dataset import DatasetAssumptionError, load_tabular_dataset


def test_load_valid_dataset(classification_csv):
    dataset = load_tabular_dataset(classification_csv)
    assert dataset.target_name == "target"
    assert dataset.feature_names == ["x1", "x2"]
    assert len(dataset) == 60


def test_missing_target_column_is_rejected(classification_csv):
    with pytest.raises(DatasetAssumptionError, match="not found"):
        load_tabular_dataset(classification_csv, target="outcome")


def test_missing_values_are_rejected(classification_frame, tmp_path):
    classification_frame.loc[0, "x1"] = None
    path = tmp_path / "data.csv"
    classification_frame.to_csv(path, index=False)
    with pytest.raises(DatasetAssumptionError, match="missing values.*x1"):
        load_tabular_dataset(path)


def test_non_numeric_columns_are_rejected(classification_frame, tmp_path):
    classification_frame["region"] = "north"
    path = tmp_path / "data.csv"
    classification_frame.to_csv(path, index=False)
    with pytest.raises(DatasetAssumptionError, match="Non-numeric.*region"):
        load_tabular_dataset(path)


def test_duplicate_rows_are_rejected(classification_frame, tmp_path):
    doubled = classification_frame.iloc[[0, 0, 1, 2]]
    path = tmp_path / "data.csv"
    doubled.to_csv(path, index=False)
    with pytest.raises(DatasetAssumptionError, match="duplicate row"):
        load_tabular_dataset(path)


def test_constant_target_is_rejected(regression_frame, tmp_path):
    regression_frame["target"] = 1.0
    path = tmp_path / "data.csv"
    regression_frame.to_csv(path, index=False)
    with pytest.raises(DatasetAssumptionError, match="constant"):
        load_tabular_dataset(path)
