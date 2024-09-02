import tarfile
from pathlib import Path

import pooch

from climate_health.datatypes import FullData
from climate_health.spatio_temporal_data.temporal_dataclass import DataSet


class MultiCountryDataSet:
    def __init__(self, data: dict[str, DataSet]):
        self._data = data

    def __getitem__(self, item):
        return self._data[item]

    @property
    def countries(self):
        return list(self._data.keys())

    def keys(self):
        return self._data.keys()

    @classmethod
    def from_tar(cls, url, dataclass=FullData):
        tar_gz_file_name = pooch.retrieve(url, known_hash=None)
        with tarfile.open(tar_gz_file_name, 'r:gz') as tar_file:
            members = tar_file.getmembers()
            extracted_files = {Path(member.name).stem: tar_file.extractfile(member) for member in members}
            print({name: ef.name for name, ef in extracted_files.items() if ef is not None})
            data = {name: DataSet.from_csv(ef, dataclass) for name, ef in extracted_files.items() if ef is not None}
        return MultiCountryDataSet(data)

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