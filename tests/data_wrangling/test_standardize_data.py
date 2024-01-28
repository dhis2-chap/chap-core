from pathlib import Path

from omnipy import StrDataset

from climate_health.data_wrangling.separated_data import load_separated_data
import pytest


# Seems to fail due to missing example data
@pytest.mark.xfail
def test_load_separated_data():
    example_data_path = Path(__file__).parent.parent.parent / 'example_data'
    data_files = (str(example_data_path / filename) for filename in
                  ('separated_disease_data.csv',
                   'separated_rain_data.csv',
                   'separated_temp_data.csv'))
    dataset = load_separated_data(data_files)
    assert isinstance(dataset, StrDataset)
    assert len(dataset) == 3
    assert tuple(dataset.keys()) == ('separated_disease_data',
                                     'separated_rain_data',
                                     'separated_temp_data')


#def test_standardize_separated_data():
#    ...
