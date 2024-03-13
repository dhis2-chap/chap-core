from typing import Protocol, Union


class TimeDelta(Protocol):
    years: int
    months: int
    days: int


class TimeStamp(Protocol):
    year: int
    month: int
    day: int

    def __le__(self, other: 'TimeStamp') -> bool:
        ...

    def __ge__(self, other: 'TimeStamp') -> bool:
        ...

    def __lt__(self, other: 'TimeStamp') -> bool:
        ...

    def __gt__(self, other: 'TimeStamp') -> bool:
        ...

    def __add__(self, other: TimeDelta) -> 'TimeStamp':
        ...


class TimePeriod:
    start_time: TimeStamp
    end_time: TimeStamp
    freqstr: str

    def __le__(self, other: Union['TimePeriod', TimeStamp]) -> bool:
        ...

    def __ge__(self, other: Union['TimePeriod', TimeStamp]) -> bool:
        ...

    def __lt__(self, other: Union['TimePeriod', TimeStamp]) -> bool:
        ...

    def __gt__(self, other: Union['TimePeriod', TimeStamp]) -> bool:
        ...

    def __add__(self, other: TimeDelta) -> 'TimePeriod':
        ...
