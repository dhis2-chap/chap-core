from chap_core.datatypes import ClimateHealthTimeSeries
import numpy as np
from chap_core.simulation.simulator import Simulator
from chap_core.time_period import PeriodRange, Year


class RandomNoiseSimulator(Simulator):
    """Simulate a random noise model."""

    def __init__(self, n_time_points: int):
        """Initialize the simulator."""
        super().__init__()
        self.n_time_points = n_time_points

    def simulate(self) -> ClimateHealthTimeSeries:
        """Simulate the model for the given parameters."""
        return ClimateHealthTimeSeries(
            time_period=PeriodRange.from_start_and_n_periods(Year(1), self.n_time_points),
            rainfall=np.random.randn(self.n_time_points),
            mean_temperature=np.random.randn(self.n_time_points),
            disease_cases=np.random.poisson(10, self.n_time_points),
        )
