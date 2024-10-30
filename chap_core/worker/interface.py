import dataclasses
from typing import TypeVar, Generic, Callable, Protocol, Any

ReturnType = TypeVar("ReturnType", covariant=True)


class Job(Generic[ReturnType], Protocol):
    @property
    def status(self) -> str: ...

    @property
    def result(self) -> ReturnType: ...

    @property
    def progress(self) -> float: ...

    def cancel(self): ...

    @property
    def is_finished(self) -> bool: ...

@dataclasses.dataclass
class SeededJob:
    status: str = 'ready'
    result: Any = None
    is_finished: bool = True
    progress: float = 1.0




class Worker(Generic[ReturnType], Protocol):
    def queue(self, func: Callable[..., ReturnType], *args, **kwargs) -> Job[ReturnType]: ...
