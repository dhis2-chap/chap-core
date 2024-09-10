from climate_health.file_io.cleaners import rwanda_data
filename = '/home/knut/Downloads/data/Malaria Cases Final.xlsx'
df = rwanda_data(filename)
print(df)