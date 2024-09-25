def get_period_id(time_period):
    if "W" in time_period:
        year, week = time_period.split("W")
        return int(year) * 53 + int(week)
    else:
        year = time_period[:4]
        month = time_period[4:]
        return int(year) * 12 + int(month)


def convert_time_period_string(row):
    if len(row) == 6 and "W" not in row:
        return f"{row[:4]}-{row[4:]}"
    return row
