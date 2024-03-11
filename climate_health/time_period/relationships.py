from . import TimePeriod, Month


def previous(period: TimePeriod) -> TimePeriod:
    if isinstance(period, Month):
        return Month(period.year-(period.month==1), (period.month - 1 -1) % 12 + 1)
    raise NotImplementedError(f"previous not implemented for {type(period)}")