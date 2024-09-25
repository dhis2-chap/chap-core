from . import TimePeriod, Month, delta_week, Year


def previous(period: TimePeriod) -> TimePeriod:
    if period.__class__.__name__ == "Month":
        return Month(period.year - (period.month == 1), (period.month - 1 - 1) % 12 + 1)
    elif period.__class__.__name__ == "Year":
        return Year(period.year - 1)
    elif period.__class__.__name__ == "Week":
        return period - delta_week

    raise NotImplementedError(f"previous not implemented for {type(period)}")
