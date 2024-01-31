from climate_health.plotting import plot_timeseries_data
from climate_health.simulation.random_noise_simulator import RandomNoiseSimulator


def test_plot_timeseries_data():
    data = RandomNoiseSimulator(100).simulate()
    plot_timeseries_data(data)
