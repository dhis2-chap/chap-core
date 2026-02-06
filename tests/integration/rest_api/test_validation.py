import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from chap_core.datatypes import create_tsdataclass
from chap_core.rest_api.v1.rest_api import app
from chap_core.rest_api.v1.routers.analytics import _validate_full_dataset
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
