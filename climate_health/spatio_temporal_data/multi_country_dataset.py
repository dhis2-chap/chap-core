from climate_health.datatypes import FullData
from climate_health.spatio_temporal_data.temporal_dataclass import DataSet


class MultiCountryDataSet:
    def __init__(self, data: dict[str, DataSet]):
        self._data = data

    def items(self):
        return self._data.items()

    @classmethod
    def from_folder(cls, folder_path, dataclass=FullData):
        csv_files = folder_path.glob('*.csv')
        data = {file.stem: DataSet.from_csv(file, dataclass) for file in csv_files}
        return MultiCountryDataSet(data)

    @property
    def period_range(self):
        return list(self._data.values())[0].period_range

    def restrict_time_period(self, time_period):
        return MultiCountryDataSet({name: data.restrict_time_period(time_period) for name, data in self._data.items()})