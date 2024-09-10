from typing import TypeVar, Protocol

ValueType = TypeVar('ValueType')
ParameterType = TypeVar('ParameterType')


class IsDistribution(Protocol[ValueType]):
    def sample(self, shape: tuple[int]) -> ValueType:
        ...

    def log_prob(self, value: ValueType) -> float:
        ...
