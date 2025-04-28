from chap_core.google_earth_engine.gee_raw import load_credentials
import pandas as pd
import pooch  # type: ignore
from dateutil.parser import parse
import pickle
from chap_core.data.open_dengue import OpenDengueDataSet
import json
from pydantic.json import pydantic_encoder
import logging

logging.basicConfig(level=logging.INFO)

country_name = 'BRAZIL'


class MonthlyDengueDataset:
    def __init__(self, local_data_path):
        self.local_data_path = local_data_path
        df = pd.read_csv(local_data_path)
        self.df = df[df['T_res'] == 'Month']
        # self._filename = pooch.retrieve(self.data_path, None)

    def retrieve_dataset(self):
        data_path = 'https://github.com/OpenDengue/master-repo/raw/main/data/releases/V1.2.2/Spatial_extract_V1_2_2.zip'
        zip_filename = pooch.retrieve(data_path, None)
        df = pd.read_csv(zip_filename, compression='zip')
        df.to_csv(self.local_data_path, index=False)

    def investigate_counts(self):
        return self.df.value_counts('adm_0_name')


if __name__ == '__main__':
    local_data_path = '/home/knut/Data/ch_data/open_dengue.csv'
    dataset = MonthlyDengueDataset(local_data_path)
    # dataset.retrieve_dataset()
    counts = dataset.investigate_counts()
#
#     df = OpenDengueDataSet()
#     base_df = OpenDengueDataSet().as_dataset('BRAZIL', spatial_resolution='Admin2', temporal_resolution='Week')
#     polygons = get_area_polygons('brazil', base_df['location'].unique(), 2)
#     credentials = load_credentials()
#     first_week,last_week = min(base_df['time_period']), max(base_df['time_period'])
#     data = gee_era5(credentials, polygons, first_week.id, last_week.id, ['temperature_2m', 'total_precipitation_sum'])
#     pickle.dump(data, open('brazil_era5_data_adm2.pkl', 'wb'))
# if False:
#     time_periods = [Week(parse(date)) for date in base_df['calendar_start_date']]
#     base_df.time_period = time_periods
#
#     brazil_data = pickle.load(open('brazil_era5_data.pkl', 'rb'))
#     df = pd.DataFrame([de.model_dump() for de in brazil_data])
#     # Make one column for each value of 'band'
#     df = df.pivot(index=('period', 'location'), columns='band', values='value')
#     # Reset the index to make the columns 'period' and 'location' into regular columns
#     df = df.reset_index()
#     # Rename the columns
#     df = df.rename(
#         columns={'period': 'time_period', 'temperature_2m': 'mean_temperature', 'total_precipitation_sum': 'rainfall'})
#
#     # df2 = pd.read_csv('brazil_weekly_data.csv')
#
