from climate_health.spatio_temporal_data.multi_country_dataset import (
    MultiCountryDataSet,
)

ISIMIP_dengue_harmonized = MultiCountryDataSet.from_tar(
    "https://github.com/dhis2/chap-core/raw/dev/example_data/full_data.tar.gz"
)
