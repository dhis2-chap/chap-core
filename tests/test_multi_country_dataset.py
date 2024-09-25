from chap_core.spatio_temporal_data.multi_country_dataset import (
    MultiCountryDataSet,
)


def test_from_tar():
    url = "https://github.com/dhis2/chap-core/raw/dev/example_data/full_data.tar.gz"
    dataset = MultiCountryDataSet.from_tar(url)
    assert "brazil" in dataset.countries
