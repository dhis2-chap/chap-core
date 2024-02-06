import numpy as np
import pandas as pd
from bionumpy.bnpdataclass import bnpdataclass

@bnpdataclass
class Period:
    ...


@bnpdataclass
class Year(Period):
    year: int

    def topandas(self):
        return pd.Series([pd.Period(year=y, freq='Y') for y in self.year])

    def __array_function__(self, func, types, args, kwargs):
        if func == np.argsort:
            return self.__argsort__()
        return super().__array_function__(func, types, args, kwargs)

    def __argsort__(self):
        return np.argsort(self.year)

@bnpdataclass
class Month(Year):
    month: int

    def __argsort__(self):
        return np.lexsort((self.month, self.year))


@bnpdataclass
class Day(Month):
    day: int

    def __argsort__(self):
        return np.lexsort((self.day, self.month, self.year))
