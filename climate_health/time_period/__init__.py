from typing import Union


class TimePeriod:
    def __init__(self):
        pass

    @classmethod
    def from_string(cls, time_string):
        split = time_string.split('-')

        if len(split) == 2:
            year, month = split
            return Month(year=year, month=month)
        elif len(split) == 3:
            year, month, day = split
            return Day(int(year), int(month), int(day))


class Month:
    def __init__(self, year: Union[int, str], month: Union[int, str]) -> None:
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


class Day:
    def __init__(self, year: Union[int, str], month: Union[int, str], day: Union[int, str]) -> None:
        self.year = int(year)
        self.month = int(month)
        self.day = int(day)



def get_number_of_days():
    return None