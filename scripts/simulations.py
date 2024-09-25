from chap_core.datatypes import ClimateHealthTimeSeries
from chap_core.plotting import plot_timeseries_data
import pandas as pd

def convert_stacked_data_to_standard(csv_file_name):
    # hacky conversion; note special separator in csv file
    data = pd.read_csv(csv_file_name, sep=";")
    data_wide = data.pivot(index="Period", columns="Data", values="Value")
    data_wide.reset_index(inplace=True)
    pd.set_option('display.max_columns', None)
    data_wide.columns = ["Time", "Rain", "Temperature", "Disease"]
    data_wide["Year"] = data_wide["Time"].str.split(" ", expand=True)[2]
    data_wide["week"] = data_wide["Time"].str.split(" ", expand=True)[0]
    data_wide["week"] = data_wide["week"].str.replace("W", "")
    data_wide["Year"] = data_wide["Year"].astype(int)
    data_wide["week"] = data_wide["week"].astype(int)
    data_wide = data_wide.sort_values(by=["Year", "week"])
    return ClimateHealthTimeSeries.from_pandas(data_wide)

def plot_real_data():
    data = convert_stacked_data_to_standard("../real_data.csv")
    fig = plot_timeseries_data(data)
    fig.show()


if __name__ == '__main__':
    plot_real_data()
