from abc import abstractmethod
from typing import Protocol

from chap_core.datatypes import ClimateData, ClimateHealthTimeSeries, HealthData


class Simulator:
    def __init__(self): ...

    @abstractmethod
    def simulate(self) -> ClimateHealthTimeSeries:
        """Simulate the model for the given parameters."""
        raise NotImplementedError()


class IsDiseaseCaseSimulator(Protocol):
    def simulate(self, climate_data: ClimateData) -> HealthData: ...
