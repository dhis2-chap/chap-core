from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

from chap_core.datatypes import ClimateHealthTimeSeries, ClimateData, HealthData
import plotly.express as px
import pandas as pd


def plot_timeseries_data(data: ClimateHealthTimeSeries) -> Figure:
    """Can be used to plot ClimateHealthTimeSeries data.
    It will create one sub-plot for each of the variables in the data. with time_period on x-axis.
    returns a plotly object that can be shown or saved."""
    df = data.topandas()
    df = pd.melt(df, id_vars=["time_period"], var_name="variable", value_name="value")
    fig = px.line(
        df,
        x="time_period",
        y="value",
        facet_row="variable",
        title="Climate Health Data",
    )
    fig.update_yaxes(matches=None)
    return fig


def plot_multiperiod(climate_data: ClimateData, health_data: HealthData, head: None | int = None) -> Figure:
    """Returns a plot of the climate and health data on the same plot. The time_period is on the x-axis.

    Parameters
    ----------
        head: int
            Number of rows to plot. If None, all rows are plotted.
    """
    climate_df = climate_data.topandas()
    climate_df = climate_df.head(head)
    climate_df.time_period = climate_df.time_period.dt.to_timestamp()
    last_month, last_year = (
        climate_df.time_period.iloc[-1].month,
        climate_df.time_period.iloc[-1].year,
    )
    cut_off = pd.Period(year=last_year, month=last_month, freq="M")

    health_df = health_data.topandas()
    cut_off_idx = (health_df.time_period == cut_off).to_list().index(True) + 1
    health_df = health_df.head(cut_off_idx)
    health_df.time_period = health_df.time_period.dt.to_timestamp()

    temperature_trace = px.line(climate_df, x="time_period", y="mean_temperature", title="Climate Health Data")
    temperature_trace.update_traces(line_color="#1E88E5")
    disease_trace = px.line(
        health_df,
        x="time_period",
        y="disease_cases",
        title="Climate Health Data",
        line_shape="vh",
    )
    disease_trace.update_traces(line_color="#D81B60")
    disease_trace.update_traces(yaxis="y2")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_traces(temperature_trace.data + disease_trace.data)
    fig.layout.xaxis.title = "Time"
    fig.layout.yaxis.title = "Temperature"
    fig.layout.yaxis2.title = "Number of disease cases"
    return fig
