import os

import pytest

from climate_health.datatypes import ClimateData
from climate_health.plotting import plot_timeseries_data
from climate_health.simulation.random_noise_simulator import RandomNoiseSimulator
from tests import EXAMPLE_DATA_PATH


def test_plot_timeseries_data():
    data = RandomNoiseSimulator(100).simulate()
    plot_timeseries_data(data)

@pytest.mark.xfail(reason="Plotting doesnt work with Period on the x-axis. Need to fix this.")
def test_plot_timeseries_data_and_write():
    filename = "test_plot_timeseries_data_and_write.png"
    data = RandomNoiseSimulator(100).simulate()
    plot_timeseries_data(data).write_image(filename)
    assert os.path.exists(filename)
    os.remove(filename)

@pytest.fixture()
def daily_climate_data():
    return ClimateData.from_csv(EXAMPLE_DATA_PATH / 'climate_data_daily.csv')

@pytest.mark.skip
def test_multiperiod_plot(daily_climate_data: ClimateData):
    monthly_health_data = create_monthly_health_data()
    plot_multiperiod(daily_climate_data, monthly_health_data)



