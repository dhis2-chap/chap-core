from typing import TypeVar, Generic, Callable, Protocol

ReturnType = TypeVar('ReturnType', covariant=True)


class Job(Generic[ReturnType], Protocol):

    @property
    def status(self) -> str:
        ...

    @property
    def result(self) -> ReturnType:
        ...


class Worker(Protocol):
    def queue(self, func: Callable[..., ReturnType], *args, **kwargs) -> Job[ReturnType]:
        ...
