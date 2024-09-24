from typing import Protocol, TypeVar, Any, Sequence

from climate_health.external.models.jax_models.prototype_distributions import (
    ParameterType,
    ValueType,
)

StateType = TypeVar("StateType")
InputType = TypeVar("InputType")


class ObservationDistribution(Protocol[StateType, InputType, ValueType]):
    def __init__(
        self, state: StateType, global_params: ParameterType, input: dict[str, Any]
    ): ...

    def sample(self, shape: tuple[int]) -> ValueType: ...

    def log_prob(self, state: StateType) -> float: ...


class StateDistribution(Protocol[StateType, ParameterType, InputType]):
    def __init__(
        self, previous_state: StateType, params: ParameterType, input: InputType
    ): ...

    def sample(self, shape: tuple[int]) -> StateType: ...

    def log_prob(self, state: StateType) -> float: ...


class InitialStateDistribution(Protocol[StateType, ParameterType]):
    def __init__(self, params: ParameterType): ...

    def sample(self, shape: tuple[int]) -> StateType: ...

    def log_prob(self, state: StateType) -> float: ...


class StateSpaceModel(Protocol[StateType, InputType, ValueType, ParameterType]):
    def __init__(self, global_params: ParameterType, input: InputType): ...

    def sample(self, shape: tuple[int]) -> Sequence[ValueType]: ...

    def log_prob(self, value: Sequence[ValueType]) -> float: ...


def state_distribution(
    previous_state: StateType, params: ParameterType, input: InputType
) -> StateDistribution[StateType]: ...
