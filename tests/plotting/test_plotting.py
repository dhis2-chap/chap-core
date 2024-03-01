import os

import numpy as np
import pytest

from climate_health.datatypes import ClimateData, HealthData
from climate_health.plotting import plot_timeseries_data
from climate_health.plotting.prediction_plot import forecast_plot
from climate_health.simulation.random_noise_simulator import RandomNoiseSimulator
from climate_health.time_period import Month
from climate_health.time_period.period_range import period_range
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


def create_monthly_health_data_for_time(daily_data):
    start_day = daily_data.time_period[0]
    start_month = Month(start_day.year, start_day.month)
    end_day = daily_data.time_period[-1]
    end_month = Month(end_day.year, end_day.month)
    return period_range(start_month, end_month, exclusive_end=False)



@pytest.mark.xfail()
def test_multiperiod_plot(daily_climate_data: ClimateData):
    monthly_range = create_monthly_health_data_for_time(daily_climate_data)
    monthly_health_data = HealthData(monthly_range, np.arange(len(monthly_range)))
    print(monthly_health_data.time_period)
    print(daily_climate_data.time_period)
    plot_multiperiod(daily_climate_data, monthly_health_data)


class MockSampler:
    def __init__(self, true_data):
        self.true_data = true_data

    def sample(self, climate_data):
        return self.true_data[:len(climate_data)].disease_cases+np.random.poisson(10, len(climate_data))

@pytest.mark.xfail
def test_forecast_plot():
    # create curve with period 12
    tp = period_range(Month(2010, 1), Month(2020, 1))
    time = np.arange(len(tp))
    real_cases = np.sin(time*2*np.pi/12)*20+100
    health_data=HealthData(tp, real_cases)
    sampler = MockSampler(health_data)
    climate_data = tp
    forecast_plot(health_data, sampler, climate_data, 100)

