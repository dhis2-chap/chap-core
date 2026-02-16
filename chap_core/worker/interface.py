import dataclasses
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

ReturnType = TypeVar("ReturnType")


class Job[ReturnType_co](Protocol):
    @property
    def status(self) -> str: ...

    @property
    def result(self) -> ReturnType_co: ...

    @property
    def progress(self) -> float: ...

    @property
    def exception_info(self) -> str: ...

    def cancel(self) -> None: ...

    @property
    def is_finished(self) -> bool: ...


@dataclasses.dataclass
class SeededJob:
    status: str = "ready"
    result: Any = None
    is_finished: bool = True
    progress: float = 1.0
    exception_info: str = ""

    def cancel(self) -> None:
        pass


class Worker[ReturnType](Protocol):
    def queue(self, func: Callable[..., ReturnType], *args, **kwargs) -> Job[ReturnType]: ...
