import pytest
from chap_core.spatio_temporal_data.multi_country_dataset import (
    MultiCountryDataSet,
)


@pytest.mark.slow
def test_from_tar():
    url = "https://raw.githubusercontent.com/dhis2-chap/chap-core/master/example_data/full_data.tar.gz"
    dataset = MultiCountryDataSet.from_tar(url)
    assert "brazil" in dataset.countries
