from pathlib import Path

import pytest

from climate_health.gluonts_adaptor.dataset import DataSetAdaptor, get_dataset, get_split_dataset
from climate_health.spatio_temporal_data.multi_country_dataset import MultiCountryDataSet
from .data_fixtures import train_data_pop, full_data

@pytest.fixture
def full_dataset():
    foldername = Path('/home/knut/Data/ch_data/full_data')
    if not foldername.exists():
        pytest.skip()
    dataset = MultiCountryDataSet.from_folder(foldername)
    return dataset

def test_to_gluonts(train_data_pop):
    dataset = DataSetAdaptor().to_gluonts(train_data_pop)
    dataset = list(dataset)
    assert len(dataset) == 2
    assert dataset[0]['target'].shape == (7,)
    assert dataset[0]['feat_dynamic_real'].shape == (7, 3)
    for i, data in enumerate(dataset):
        assert data['feat_static_cat'] == [i]

def test_get_dataset():
    dataset = list(get_dataset('laos_full_data'))
    for data in dataset:
        print(data)

def test_get_split_dataset():
    train, test = get_split_dataset('laos_full_data', n_periods=6)
    first_train = list(train)[0]
    first_test = list(test)[0]
    print(first_train.keys())
    print(first_test.keys())
    assert len(first_test['target']) == len(first_train['target'])+6

def test_full_data(full_dataset):
    print(list(DataSetAdaptor.to_gluonts_multicountry(full_dataset)))




