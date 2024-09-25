from chap_core.datatypes import ClimateHealthTimeSeries
from chap_core.simulation.random_noise_simulator import RandomNoiseSimulator


def test_random_noise_simulator():
    my_simulator = RandomNoiseSimulator(10)
    data = my_simulator.simulate()
    assert len(data.time_period) == 10
    assert isinstance(data, ClimateHealthTimeSeries)
