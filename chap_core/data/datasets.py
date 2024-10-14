from chap_core.spatio_temporal_data.multi_country_dataset import LazyMultiCountryDataSet

ISIMIP_dengue_harmonized = LazyMultiCountryDataSet(
    "https://github.com/dhis2/chap-core/raw/dev/example_data/full_data.tar.gz"
)
