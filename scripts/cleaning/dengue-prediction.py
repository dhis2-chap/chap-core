import pandas as pd

from chap_core.datatypes import FullData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

data_path = '../../example_data/dengue_input_from_source_v10.csv'
df = pd.read_csv(data_path)
print(df.columns)
df = df.rename(columns={
    'nome_bairro': 'location',
    'dengue_diagnosis': 'disease_cases',
    'ano': 'year',
    'mes': 'month',
    'precipitacao (mm)': 'rainfall',
    'temperatura (°C)': 'mean_temperature',
    'Populacao': 'population'})
df['time_period'] = df['year'].astype(str) + '-' + df['month'].apply(lambda x: str(x).zfill(2))
df.sort_values(['location', 'time_period'], inplace=True)

dataset = DataSet.from_pandas(df, FullData)
dataset.to_csv('../../example_data/dengue_prediction_data.csv')


#df.rename(columns={'