from typing import Protocol

from climate_health.datatypes import ClimateHealthTimeSeries


class Simulator:
    def __init__(self):
        ...

    def simulate(self) -> ClimateHealthTimeSeries:
        """Simulate the model for the given parameters."""
        ...



