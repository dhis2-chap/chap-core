import pandas as pd
import numpy as np
import pytest
from bionumpy.util.testing import assert_bnpdataclass_equal
from chap_core.datatypes import ClimateHealthTimeSeries, HealthData, Samples, HealthPopulationData, SimpleClimateData, \
    FullData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.testing.testing import assert_tsdataclass_equal
from chap_core.time_period import PeriodRange


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
    true_periods = PeriodRange.from_strings(["2010", "2011", "2012"])
    # bnp_ragged_array = true_periods
    # assert ts.time_period == bnp_ragged_array
    assert all(ts.time_period == true_periods)
    # assert_bnpdataclass_equal(ts.time_period, bnp_ragged_array)
    np.testing.assert_array_equal(ts.rainfall, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(ts.mean_temperature, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(ts.disease_cases, np.array([1, 2, 3]))


def test_climate_health_time_series_to_csv(tmp_path):
    """Test the to_csv method."""
    ts = ClimateHealthTimeSeries(
        time_period=PeriodRange.from_strings(["2010", "2011", "2012"]),
        rainfall=np.array([1.0, 2.0, 3.0]),
        mean_temperature=np.array([1.0, 2.0, 3.0]),
        disease_cases=np.array([1, 2, 3]),
    )
    csv_file = tmp_path / "test.csv"
    ts.to_csv(csv_file)
    ts2 = ClimateHealthTimeSeries.from_csv(csv_file)
    assert ts == ts2
    # assert ts == ts2


@pytest.fixture()
def dataset_with_missing(data_path):
    return pd.read_csv(data_path / "laos_pulled_data.csv")


# @pytest.mark.skip('Must be fixed!!!!!!')
def test_dataset_with_missing(dataset_with_missing):
    health_data = DataSet.from_pandas(
        dataset_with_missing, dataclass=HealthData, fill_missing=True
    )
    start = health_data.start_timestamp
    end = health_data.end_timestamp
    for location, data in health_data.items():
        # assert data.start_timestamp == start
        assert data.end_timestamp == end


@pytest.fixture()
def samples():
    time_period = PeriodRange.from_strings(["2010", "2011", "2012"])
    return Samples(time_period=time_period, samples=np.random.rand(3, 10))


def test_samples(samples, tmp_path):
    path = tmp_path / "samples.csv"
    samples.to_csv(path)
    samples2 = Samples.from_csv(path)
    assert_tsdataclass_equal(samples, samples2)



@pytest.fixture()
def health_population_data():
    return HealthPopulationData(PeriodRange.from_strings(['2010-01', '2010-02', '2010-03']), [1, 2, 3], [10, 20, 30])


@pytest.fixture()
def climate_data():
    return SimpleClimateData(PeriodRange.from_strings(['2010-01', '2010-02', '2010-03']),
                             np.array([0.5, 0.10, 0.20]),
                             np.array([20, 21, 22]))


def test_serialize_round_trip(climate_data):
    s = climate_data.model_dump()
    climate_data2 = SimpleClimateData.from_dict(s)



def test_merge(health_population_data, climate_data):
    merged = health_population_data.merge(climate_data, FullData)
    assert np.all(merged.rainfall == climate_data.rainfall)


def test_merge_raises(health_population_data, climate_data):
    health_population_data = health_population_data[:2]
    with pytest.raises(ValueError):
        merged = health_population_data.merge(climate_data, ClimateHealthTimeSeries)
