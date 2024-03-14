from climate_health.datatypes import ClimateHealthTimeSeries
from .data_fixtures import full_data
from tempfile import NamedTemporaryFile


def test_to_from_csv(full_data):
    with NamedTemporaryFile() as f:
        full_data.to_csv(f.name)
        full_data2 = full_data.from_csv(f.name, ClimateHealthTimeSeries)
        assert len(full_data2.data()) == len(full_data.data())

