import numpy as np
import pandas as pd
from bionumpy.bnpdataclass import bnpdataclass


@bnpdataclass
class Period: ...


@bnpdataclass
class Year(Period):
    year: int

    def topandas(self):
        return pd.Series([pd.Period(year=y, freq="Y") for y in self.year])

    def __array_function__(self, func, types, args, kwargs):
        if func == np.argsort:
            return self.argsort()
        return super().__array_function__(func, types, args, kwargs)

    def argsort(self):
        return np.argsort(self.year)

    def __le__(self, other):
        return self.year <= other.year


@bnpdataclass
class Month(Year):
    month: int

    def topandas(self):
        return pd.Series(
            [
                pd.Period(year=y, month=m, freq="M")
                for y, m in zip(self.year, self.month)
            ]
        )

    def argsort(self):
        return np.lexsort((self.month, self.year))

    def __le__(self, other):
        return np.where(
            self.year == other.year, self.month <= other.month, self.year < other.year
        )

    def __ge__(self, other):
        return np.where(
            self.year == other.year, self.month >= other.month, self.year > other.year
        )


@bnpdataclass
class Day(Month):
    day: int

    def argsort(self):
        return np.lexsort((self.day, self.month, self.year))

    def topandas(self):
        return pd.Series(
            [
                pd.Period(year=y, month=m, day=d, freq="D")
                for y, m, d in zip(self.year, self.month, self.day)
            ]
        )


class Week(Year):
    week: int

    def argsort(self):
        return np.lexsort((self.week, self.year))

    def topandas(self):
        return pd.Series([f"{y}-W{w}" for y, w in zip(self.year, self.week)])
