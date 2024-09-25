from typing import Protocol, Union


class IsTimeDelta(Protocol):
    years: int
    months: int
    days: int


class IsTimeStamp(Protocol):
    year: int
    month: int
    day: int

    def __le__(self, other: "IsTimeStamp") -> bool: ...

    def __ge__(self, other: "IsTimeStamp") -> bool: ...

    def __lt__(self, other: "IsTimeStamp") -> bool: ...

    def __gt__(self, other: "IsTimeStamp") -> bool: ...

    def __add__(self, other: IsTimeDelta) -> "IsTimeStamp": ...


class IsTimePeriod(Protocol):
    start_time: IsTimeStamp
    end_time: IsTimeStamp
    freqstr: str

    def __le__(self, other: Union["IsTimePeriod", IsTimeStamp]) -> bool: ...

    def __ge__(self, other: Union["IsTimePeriod", IsTimeStamp]) -> bool: ...

    def __lt__(self, other: Union["IsTimePeriod", IsTimeStamp]) -> bool: ...

    def __gt__(self, other: Union["IsTimePeriod", IsTimeStamp]) -> bool: ...

    def __add__(self, other: IsTimeDelta) -> "IsTimePeriod": ...


class IsPeriodRange(Protocol): ...
