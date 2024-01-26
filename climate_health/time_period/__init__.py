class TimePeriod:
    def __init__(self):
        pass
    @classmethod
    def from_string(cls, time_string):
        year, month = time_string.split('-')
        return Month(year=year, month=month)


class Month:
    def __init__(self, year: int|str, month: int|str) -> None:
        """
        :param year:
        :param month: Starting from 1
        """
        self.year = int(year)
        self.month = int(month)


    def __str__(self) -> str:
        dict_month = {
            1: 'January',
            2: 'February',
            3: 'March',
            4: 'April',
            5: 'May',
            6: 'June',
            7: 'July',
            8: 'August',
            9: 'September',
            10: 'October',
            11: 'November',
            12: 'December',
        }
        return f'{dict_month[self.month]} {self.year}'