import pandas as pd
import numpy as np
import bionumpy as bnp
from bionumpy.util.testing import assert_bnpdataclass_equal
from climate_health.datatypes import ClimateHealthTimeSeries
from climate_health.time_period.dataclasses import Year


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
    true_periods = bnp.as_encoded_array(["2010", "2011", "2012"])
    true_periods = Year([2010, 2011, 2012])
    bnp_ragged_array = true_periods
    # assert ts.time_period == bnp_ragged_array
    assert_bnpdataclass_equal(ts.time_period, bnp_ragged_array)
    np.testing.assert_array_equal(ts.rainfall, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(ts.mean_temperature, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(ts.disease_cases, np.array([1, 2, 3]))


def test_climate_health_time_series_to_csv(tmp_path):
    """Test the to_csv method."""
    ts = ClimateHealthTimeSeries(
        time_period=Year([2010, 2011, 2012]),
        rainfall=np.array([1.0, 2.0, 3.0]),
        mean_temperature=np.array([1.0, 2.0, 3.0]),
        disease_cases=np.array([1, 2, 3]),
    )
    csv_file = tmp_path / "test.csv"
    ts.to_csv(csv_file)
    ts2 = ClimateHealthTimeSeries.from_csv(csv_file)
    assert_bnpdataclass_equal(ts, ts2)
    #assert ts == ts2