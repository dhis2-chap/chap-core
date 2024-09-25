from chap_core.datatypes import ClimateHealthTimeSeries
import numpy as np
from chap_core.simulation.simulator import Simulator
from chap_core.time_period.dataclasses import Year


class RandomNoiseSimulator(Simulator):
    """Simulate a random noise model."""

    def __init__(self, n_time_points: int):
        """Initialize the simulator."""
        super().__init__()
        self.n_time_points = n_time_points

    def simulate(self) -> ClimateHealthTimeSeries:
        """Simulate the model for the given parameters."""
        return ClimateHealthTimeSeries(
            time_period=Year([i for i in range(1, self.n_time_points + 1)]),
            rainfall=np.random.randn(self.n_time_points),
            mean_temperature=np.random.randn(self.n_time_points),
            disease_cases=np.random.poisson(10, self.n_time_points),
        )
