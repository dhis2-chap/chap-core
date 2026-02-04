from . import Month, TimePeriod, Year, delta_week


def previous(period: TimePeriod) -> TimePeriod:
    if period.__class__.__name__ == "Month":
        return Month(period.year - (period.month == 1), (period.month - 1 - 1) % 12 + 1)
    elif period.__class__.__name__ == "Year":
        return Year(period.year - 1)
    elif period.__class__.__name__ == "Week":
        result: TimePeriod = period - delta_week
        return result

    raise NotImplementedError(f"previous not implemented for {type(period)}")
