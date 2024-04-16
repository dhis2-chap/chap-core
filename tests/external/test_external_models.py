import logging
from pathlib import Path

import pandas as pd
import pytest
import yaml

from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
from climate_health.datatypes import ClimateHealthTimeSeries

logging.basicConfig(level=logging.INFO)
from climate_health.external.external_model import get_model_from_yaml_file
from ..data_fixtures import full_data, train_data, future_climate_data
from climate_health.util import conda_available


@pytest.mark.skipif(not conda_available(), reason='requires conda')
def test_r_model_from_folder(models_path, train_data, future_climate_data):
    yaml = models_path / 'testmodel' / 'config.yml'
    model = get_model_from_yaml_file(yaml)
    model.setup()
    model.train(train_data)
    with pytest.raises(ValueError):
        model.predict(future_climate_data)


def test_python_model_from_folder(models_path, train_data, future_climate_data):
    yaml = models_path / 'naive_python_model' / 'config.yml'
    model = get_model_from_yaml_file(yaml)
    model.train(train_data)
    results = model.predict(future_climate_data)
    assert results is not None


def get_dataset_from_yaml(yaml_path: Path):
    specs = yaml.load(yaml_path.read_text(), Loader=yaml.FullLoader)
    if 'demo_data' in specs:
        df = pd.read_csv(yaml_path.parent / specs['demo_data'])
    if 'demo_data_adapter' in specs:
        for to_name, from_name in specs['demo_data_adapter'].items():
            if '{' in from_name:
                new_col = [from_name.format(**df.iloc[i].to_dict()) for i in range(len(df))]
                df[to_name] = new_col
            else:
                df[to_name] = df[from_name]

    return SpatioTemporalDict.from_pandas(df, ClimateHealthTimeSeries)

@pytest.mark.parametrize('model_directory', ['ewars_Plus'])
def test_all_external_models_acceptance(model_directory, models_path, train_data, future_climate_data):
    """Only tests that the model can be initiated and that train and predict
    can be called without anything failing"""
    yaml_path = models_path / model_directory / 'config.yml'
    model = get_model_from_yaml_file(yaml_path)
    # with open(yaml_file, 'r') as file:
    train_data = get_dataset_from_yaml(yaml_path)
    model.setup()
    model.train(train_data)
    try:
        results = model.predict(future_climate_data)
        assert results is not None
    except ValueError:
        # This is expected for some models
        pass
