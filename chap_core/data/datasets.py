from chap_core.spatio_temporal_data.multi_country_dataset import LazyMultiCountryDataSet

ISIMIP_dengue_harmonized = LazyMultiCountryDataSet(
    "https://raw.githubusercontent.com/dhis2-chap/chap-core/master/example_data/full_data.tar.gz"
)
