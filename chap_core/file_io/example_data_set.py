from pathlib import Path
from typing import Literal

from chap_core.datatypes import ClimateHealthTimeSeries, FullData
from chap_core.spatio_temporal_data.multi_country_dataset import (
    MultiCountryDataSet,
)
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


class ExampleDataSet:
    base_path = Path(__file__).parent.parent.parent / "example_data"

    def __init__(self, name, dataclass=ClimateHealthTimeSeries):
        self._name = Path(name)
        self._dataclass = dataclass

    def filepath(self):
        return self.base_path / self._name.with_suffix(".csv")

    def load(self) -> DataSet:
        filename = self._name.with_suffix(".csv")
        filepath = self.base_path / filename
        return DataSet.from_csv(filepath, dataclass=self._dataclass)


class RemoteExampleDataSet:
    def __init__(self, url: str):
        self._url = url

    def load(self) -> DataSet:
        return MultiCountryDataSet.from_tar(self._url)


class LocalDataSet(ExampleDataSet):
    base_path = Path(__file__).parent.parent.parent.parent.parent / "Data"


dataset_names = [
    "hydro_met_subset",
    "hydromet_clean",
    "hydromet_10",
    "hydromet_5_filtered",
]
local_datasets = ["laos_full_data", "uganda_data"]
remote_datasets = {
    "ISIMIP_dengue_harmonized": "https://github.com/dhis2/chap-core/raw/dev/example_data/full_data.tar.gz"
}
DataSetType = Literal[tuple(dataset_names) + tuple(local_datasets) + tuple(remote_datasets.keys())]
datasets: dict[str, ExampleDataSet] = {
    name: ExampleDataSet(name) if name != "hydromet_5_filtered" else ExampleDataSet(name, FullData)
    for name in dataset_names
} | {name: LocalDataSet(name, FullData) for name in local_datasets}
for name, url in remote_datasets.items():
    datasets[name] = RemoteExampleDataSet(url)
