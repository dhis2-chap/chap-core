from pathlib import Path
from typing import Literal

from climate_health.datatypes import ClimateHealthTimeSeries, FullData
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict


class ExampleDataSet:
    base_path = Path(__file__).parent.parent.parent / 'example_data'

    def __init__(self, name, dataclass=ClimateHealthTimeSeries):
        self._name = Path(name)
        self._dataclass = dataclass

    def load(self):
        filename = self._name.with_suffix('.csv')
        filepath = self.base_path / filename
        return SpatioTemporalDict.from_csv(filepath, dataclass=self._dataclass)


class LocalDataSet(ExampleDataSet):
    base_path = Path(__file__).parent.parent.parent.parent.parent / 'Data'


dataset_names = ['hydro_met_subset', 'hydromet_clean', 'hydromet_10', 'hydromet_5_filtered']
local_datasets = ['laos_full_data', 'uganda_data']
DataSetType = Literal[tuple(dataset_names)+tuple(local_datasets)]
datasets = {name: ExampleDataSet(name) if name != 'hydromet_5_filtered' else ExampleDataSet(name, FullData) for name in
            dataset_names} | {name: LocalDataSet(name, FullData) for name in local_datasets}
