from pathlib import Path

import pytest

from climate_health.assessment.dataset_splitting import train_test_split
from climate_health.data.gluonts_adaptor.dataset import DataSetAdaptor, get_dataset, get_split_dataset
from climate_health.datatypes import FullData, remove_field
from climate_health.spatio_temporal_data.multi_country_dataset import MultiCountryDataSet
from climate_health.file_io.example_data_set import datasets
from climate_health.spatio_temporal_data.temporal_dataclass import DataSet
from .data_fixtures import train_data_pop, full_data
from climate_health.data.datasets import ISIMIP_dengue_harmonized


@pytest.fixture
def full_dataset():
    dataset = ISIMIP_dengue_harmonized
    return dataset


@pytest.fixture
def gluonts_vietnam_dataset():
    dataset = ISIMIP_dengue_harmonized['vietnam']
    return DataSetAdaptor.from_dataset(dataset)


def test_to_dataset(gluonts_vietnam_dataset):
    dataset = DataSetAdaptor.to_dataset(gluonts_vietnam_dataset, FullData)
    assert isinstance(dataset, DataSet)
    assert len(dataset.keys()) > 3


def test_to_testinstances(train_data_pop: DataSet):
    train, test = train_test_split(train_data_pop, prediction_start_period=train_data_pop.period_range[-3])
    ds = DataSetAdaptor().to_gluonts_testinstances(train, test.remove_field('disease_cases'), 3)
    print(list(ds))


def test_to_gluonts(train_data_pop):
    dataset = DataSetAdaptor().to_gluonts(train_data_pop)
    dataset = list(dataset)
    assert len(dataset) == 2
    assert dataset[0]['target'].shape == (7,)
    assert dataset[0]['feat_dynamic_real'].shape == (3, 7)
    for i, data in enumerate(dataset):
        assert data['feat_static_cat'] == [i]


@pytest.fixture()
def laos_full_data():
    dataset = datasets['laos_full_data']
    if not dataset.filepath().exists():
        pytest.skip()
    return 'laos_full_data'


def test_get_dataset(laos_full_data):
    dataset = list(get_dataset(laos_full_data))
    for data in dataset:
        print(data)


def test_get_split_dataset(laos_full_data):
    train, test = get_split_dataset(laos_full_data, n_periods=6)
    first_train = list(train)[0]
    first_test = list(test)[0]
    print(first_train.keys())
    print(first_test.keys())
    assert len(first_test['target']) == len(first_train['target']) + 6


def test_full_data(full_dataset):
    print(list(DataSetAdaptor.to_gluonts_multicountry(full_dataset)))
