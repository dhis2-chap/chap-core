import pandas as pd

filepath = '~/Downloads/Temporal_extract_V1_2_2.csv'
df = pd.read_csv(filepath)
print(df['S_res'].value_counts())
print(df['T_res'].value_counts())

spatial_filepath = '~/Downloads/Spatial_extract_V1_2_2.csv'
dfS = pd.read_csv(spatial_filepath)
print(dfS['S_res'].value_counts())
print(dfS['T_res'].value_counts())
