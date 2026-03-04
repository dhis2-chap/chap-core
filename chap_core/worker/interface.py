from collections.abc import Callable
from typing import Protocol, TypeVar

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


class Worker[ReturnType](Protocol):
    def queue(self, func: Callable[..., ReturnType], *args, **kwargs) -> Job[ReturnType]: ...
