import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from chap_core.datatypes import create_tsdataclass
from chap_core.rest_api.app import app
from chap_core.api_types import BacktestParams
from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.rest_api.v1.routers.analytics import (
    _filter_dataset_by_locations,
    _find_locations_with_target_data,
    _validate_full_dataset,
)
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import PeriodRange

client = TestClient(app)


def test_(): ...


def test_validate_full_dataset_error_uses_camel_case():
    dataclass = create_tsdataclass(["disease_cases", "rainfall"])
    period_range = PeriodRange.from_strings(["2020-01", "2020-02", "2020-03"])
    data = dataclass(period_range, np.array([1.0, 2.0, 3.0]), np.array([np.nan, np.nan, np.nan]))
    dataset = DataSet({"loc1": data})

    with pytest.raises(HTTPException) as exc_info:
        _validate_full_dataset(["disease_cases", "rainfall"], dataset)

    detail: dict = exc_info.value.detail  # type: ignore[assignment]
    rejected_entry = detail["rejected"][0]
    assert rejected_entry["orgUnit"] == "loc1"
    assert rejected_entry["featureName"] == "rainfall"
    assert rejected_entry["timePeriods"] == ["202001", "202002", "202003"]
    assert "org_unit" not in rejected_entry
    assert "feature_name" not in rejected_entry
    assert "time_periods" not in rejected_entry


def test_validate_target_in_first_train_split_rejects_all_nan_location():
    dataclass = create_tsdataclass(["disease_cases", "rainfall"])
    period_range = PeriodRange.from_strings(
        ["2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06", "2020-07", "2020-08", "2020-09", "2020-10"]
    )
    # loc1: has data throughout
    data_good = dataclass(
        period_range,
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        np.array([1.0] * 10),
    )
    # loc2: all NaN disease_cases in the first train split (NaN for first 4 periods)
    data_bad = dataclass(
        period_range,
        np.array([np.nan, np.nan, np.nan, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        np.array([1.0] * 10),
    )
    dataset = DataSet({"loc1": data_good, "loc2": data_bad})

    # n_periods=3, n_splits=2, stride=1 -> first train split ends at index -(3 + 1*1 + 1) = -5
    # i.e. first train split = periods 2020-01 to 2020-06 (inclusive)
    # loc2 has NaN for 2020-01 to 2020-04 but has 5.0 at 2020-05, so it should pass
    params = BacktestParams(n_periods=3, n_splits=2, stride=1)
    train_set, _ = train_test_generator(dataset, params.n_periods, params.n_splits, stride=params.stride)
    locations_to_keep, rejected = _find_locations_with_target_data(train_set)
    filtered = _filter_dataset_by_locations(dataset, locations_to_keep)
    assert len(rejected) == 0
    assert set(filtered.locations()) == {"loc1", "loc2"}


def test_validate_target_in_first_train_split_rejects_when_no_data():
    dataclass = create_tsdataclass(["disease_cases", "rainfall"])
    period_range = PeriodRange.from_strings(
        ["2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06", "2020-07", "2020-08", "2020-09", "2020-10"]
    )
    data_good = dataclass(
        period_range,
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        np.array([1.0] * 10),
    )
    # loc2: all NaN disease_cases in the first train split (first 6 periods all NaN)
    data_bad = dataclass(
        period_range,
        np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7.0, 8.0, 9.0, 10.0]),
        np.array([1.0] * 10),
    )
    dataset = DataSet({"loc1": data_good, "loc2": data_bad})

    # first train split = periods 2020-01 to 2020-06 -> loc2 is all NaN there
    params = BacktestParams(n_periods=3, n_splits=2, stride=1)
    train_set, _ = train_test_generator(dataset, params.n_periods, params.n_splits, stride=params.stride)
    locations_to_keep, rejected = _find_locations_with_target_data(train_set)
    filtered = _filter_dataset_by_locations(dataset, locations_to_keep)
    assert len(rejected) == 1
    assert rejected[0].org_unit == "loc2"
    assert "loc2" not in filtered.locations()
    assert "loc1" in filtered.locations()
