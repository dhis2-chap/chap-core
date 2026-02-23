"""
Script for plotting generated importance weighting with LIME
"""
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Tuple, Dict



def parse_coefficients(
    coefficients: List[Tuple[str, float]],
) -> Tuple[Dict[str, Dict], Dict]:
    temp_columns = {}
    static_columns = {}

    for (name, value) in coefficients:
        if "_lag_" in name:
            base, lag = name.rsplit("_lag_", 1)
            if base not in temp_columns.keys():
                temp_columns[base] = {}
            temp_columns[base][int(lag)] = value
        elif "_fut_" in name:
            base, lag = name.rsplit("_fut_", 1)
            if base not in temp_columns.keys():
                temp_columns[base] = {}
            temp_columns[base][int(lag)*-1] = value  # Future values indexed as negative
        else:
            static_columns[name] = value
    
    return temp_columns, static_columns



def plot_importance(
    coefficients: List[Tuple[str, float]],
    hist_df: pd.DataFrame,
    fut_df: pd.DataFrame,
    segment_indices: Dict[str, List]
):
    temp_columns, static_columns = parse_coefficients(coefficients)
    max_temp = max(
        (abs(v) for inner in temp_columns.values() for v in inner.values()),
        default=0.0
    )
    max_static = max((abs(v) for v in static_columns.values()), default=0.0)
    max_val = max(max_temp, max_static, 1.0)  # Colour intensity relative

    num_temp_columns = len(temp_columns.keys())

    # Define figures and axes
    fig, axes = plt.subplots(
        nrows=num_temp_columns, ncols=1,
        sharex=True, # Makes the plots share the same x axos
        figsize=(12, 8),
        constrained_layout=True
    )

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=-max_val, vmax=max_val)

    # Plot data historic and future
    for i, col in enumerate(temp_columns.keys()):  # TODO: Again, is the df sorted here?
        y = hist_df[col].tolist()
        y_fut = fut_df[col].tolist()
        x = range(len(y))
        x_fut = range(len(y), len(y) + len(y_fut))
        axes[i].plot(x, y)
        axes[i].plot(x_fut, y_fut, color="orange")
        axes[i].set_title(f"Variable: {col}")
        axes[i].set_xlabel("Time steps")

        # Plot vertical bars at segment indices
        for lag, (startx, endx) in segment_indices[col].items():
            axes[i].axvline(
                x=endx,
                color='darkgray',
                linestyle='--',
                linewidth=1.5
            )

            # Colour background according to importance weighting
            value = temp_columns[col][lag]
            color = cmap(norm(value))

            axes[i].axvspan(
                startx, endx,
                facecolor=color,
                alpha=0.3,
                linewidth=0,
                zorder=0
            )


    plt.show()


# STILL TODO:
## Plot future segments
## Plot static variables
## Is plot a good way to present this info anyway?
## Could do some counterfactual equidistant spacing?