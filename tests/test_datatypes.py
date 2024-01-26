import pandas as pd
import numpy as np
import bionumpy as bnp

from climate_health.datatypes import ClimateHealthTimeSeries


def test_climate_health_time_series_from_csv(tmp_path):
    """Test the from_csv method."""
    data = pd.DataFrame(
        {
            "time_period": ["2010", "2011", "2012"],
            "rainfall": [1.0, 2.0, 3.0],
            "mean_temperature": [1.0, 2.0, 3.0],
            "disease_cases": [1, 2, 3],
        }
    )
    csv_file = tmp_path / "test.csv"
    data.to_csv(csv_file, index=False)
    ts = ClimateHealthTimeSeries.from_csv(csv_file)
    bnp_ragged_array = bnp.as_encoded_array(["2010", "2011", "2012"])
    assert ts.time_period == bnp_ragged_array
    np.testing.assert_array_equal(ts.rainfall, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(ts.mean_temperature, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(ts.disease_cases, np.array([1, 2, 3]))


def test_climate_health_time_series_to_csv(tmp_path):
    """Test the to_csv method."""
    ts = ClimateHealthTimeSeries(
        time_period=["2010", "2011", "2012"],
        rainfall=np.array([1.0, 2.0, 3.0]),
        mean_temperature=np.array([1.0, 2.0, 3.0]),
        disease_cases=np.array([1, 2, 3]),
    )
    csv_file = tmp_path / "test.csv"
    ts.to_csv(csv_file)
    data = pd.read_csv(csv_file, dtype={'time_period': str})
    assert data.time_period.tolist() == ["2010", "2011", "2012"]
    assert data.rainfall.tolist() == [1.0, 2.0, 3.0]
    assert data.mean_temperature.tolist() == [1.0, 2.0, 3.0]
    assert data.disease_cases.tolist() == [1, 2, 3]