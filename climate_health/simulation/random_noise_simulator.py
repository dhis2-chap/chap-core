from climate_health.datatypes import ClimateHealthTimeSeries
import numpy as np


class RandomNoiseSimulator(Simulator):
    """Simulate a random noise model."""

    def __init__(self, n_time_points: int):
        """Initialize the simulator."""
        super().__init__()
        self.n_time_points = n_time_points

    def simulate(self) -> ClimateHealthTimeSeries:
        """Simulate the model for the given parameters."""
        return ClimateHealthTimeSeries(
            time_period=[],
            rainfall=np.random.randn(self.n_time_points),
            mean_temperature=[]
        )