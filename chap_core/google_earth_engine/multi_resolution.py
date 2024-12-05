import dataclasses

from npstructures import RaggedArray

from chap_core.datatypes import HealthPopulationData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from .gee_era5 import Era5LandGoogleEarthEngine

def pack_daily_data(df, periods, dataclass):
    """
    Pack daily data into a list of data elements
    """
    n_days_in_periods  = [period.n_days for period in periods]
    field_names = [field.name for field in dataclasses.fields(dataclass) if field.name not in {'time_period',}]
    ds = {}
    for location, group in df.groupby('id'):
        group = group.sort_values('date')
        assert len(group) == sum(n_days_in_periods)
        data_dict = {}
        for field in field_names:
            data_dict[field] = RaggedArray(group[field].values, n_days_in_periods).as_padded_matrix()
        ds[location] = dataclass(time_period=periods, **data_dict)
    return DataSet(ds)

def harmonize_with_daily_data(health_dataset: DataSet[HealthPopulationData], dataclass, new_dataclass):
    """
    Harmonize daily data into a list of data elements
    """
    era5_land_gee = Era5LandGoogleEarthEngine()
    data = era5_land_gee.get_daily_data(
        health_dataset.polygons.model_dump(),
        health_dataset.period_range,
    )
    packed_data = pack_daily_data(data, health_dataset.period_range, dataclass)
    return health_dataset.merge(packed_data, new_dataclass)
