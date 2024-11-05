from chap_core.climate_data.harmonization import harmonize_health_dataset
from chap_core.datatypes import HealthData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

country_name = 'bolivia'.upper()
spatial_resolution='Admin1'
dataset = DataSet.from_csv(f'{country_name}_weekly_cases_{spatial_resolution}.csv', HealthData)
harmonized = harmonize_health_dataset(dataset, country_name.lower(), get_climate=True)