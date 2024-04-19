from pathlib import Path
from typing import Literal

from climate_health.datatypes import ClimateHealthTimeSeries
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict


class ExampleDataSet:
    def __init__(self, name, dataclass=ClimateHealthTimeSeries):
        self._name = Path(name)
        self._dataclass = dataclass

    def load(self):
        filename = self._name.with_suffix('.csv')
        filepath = Path(__file__).parent.parent.parent / 'example_data' / filename
        return SpatioTemporalDict.from_csv(filepath, dataclass=self._dataclass)



dataset_names = ['hydro_met_subset', 'hydromet_clean', 'hydromet_10']
DataSetType = Literal[tuple(dataset_names)]
datasets = {name: ExampleDataSet(name) for name in dataset_names}