from typing import Protocol

from chap_core.datatypes import ClimateHealthTimeSeries, ClimateData, HealthData


class Simulator:
    def __init__(self): ...

    def simulate(self) -> ClimateHealthTimeSeries:
        """Simulate the model for the given parameters."""
        ...


class IsDiseaseCaseSimulator(Protocol):
    def simulate(self, climate_data: ClimateData) -> HealthData: ...



