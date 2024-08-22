from climate_health.gluonts_adaptor.dataset import DataSetAdaptor
from .data_fixtures import train_data_pop, full_data


def test_to_gluonts(train_data_pop):
    dataset = DataSetAdaptor().to_gluonts(train_data_pop)
    dataset = list(dataset)
    assert len(dataset) == 2
    assert dataset[0]['target'].shape == (7,)


