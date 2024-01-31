import pandas as pd
import numpy as np
import bionumpy as bnp

from climate_health.datatypes import ClimateHealthTimeSeries


def test_climate_health_time_series_from_csv(tmp_path):
    """Test the from_csv method."""
    data = pd.DataFrame(
        {
            "Time": ["2010", "2011", "2012"],
            "Rain": [1.0, 2.0, 3.0],
            "Temperature": [1.0, 2.0, 3.0],
            "Disease": [1, 2, 3],
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
        time_period=bnp.as_encoded_array(["2010", "2011", "2012"]),
        rainfall=np.array([1.0, 2.0, 3.0]),
        mean_temperature=np.array([1.0, 2.0, 3.0]),
        disease_cases=np.array([1, 2, 3]),
    )
    csv_file = tmp_path / "test.csv"
    ts.to_csv(csv_file)
    ts2 = ClimateHealthTimeSeries.from_csv(csv_file)
    assert ts == ts2