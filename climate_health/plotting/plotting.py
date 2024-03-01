from plotly.graph_objs import Figure

from climate_health.datatypes import ClimateHealthTimeSeries
import plotly.express as px
import pandas as pd


def plot_timeseries_data(data: ClimateHealthTimeSeries) -> Figure:
    """Can be used to plot ClimateHealthTimeSeries data.
    It will create one sub-plot for each of the variables in the data. with time_period on x-axis.
    returns a plotly object that can be shown or saved."""
    df = data.topandas()
    df = pd.melt(df, id_vars=["time_period"], var_name="variable", value_name="value")
    fig = px.line(df, x="time_period", y="value", facet_row="variable", title="Climate Health Data")
    fig.update_yaxes(matches=None)
    return fig
