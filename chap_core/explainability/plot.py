"""
Script for plotting generated importance weighting with LIME
"""


import matplotlib.colors as mcolors
import pandas as pd
from matplotlib import pyplot as plt


def parse_coefficients(
    coefficients: list[tuple[str, float]],
) -> tuple[dict[str, dict], dict]:
    temp_columns = {}
    static_columns = {}

    for name, value in coefficients:
        if "_lag_" in name:
            base, lag = name.rsplit("_lag_", 1)
            if base not in temp_columns:
                temp_columns[base] = {}
            temp_columns[base][int(lag)] = value
        elif "_fut_" in name:
            base, lag = name.rsplit("_fut_", 1)
            if base not in temp_columns:
                temp_columns[base] = {}
            temp_columns[base][int(lag) * -1] = value  # Future values indexed as negative
        else:
            static_columns[name] = value

    return temp_columns, static_columns


def plot_importance(
    coefficients: list[tuple[str, float]], hist_df: pd.DataFrame, fut_df: pd.DataFrame, segment_indices: dict[str, list]
):
    temp_columns, static_columns = parse_coefficients(coefficients)
    max_temp = max((abs(v) for inner in temp_columns.values() for v in inner.values()), default=0.0)
    max_static = max((abs(v) for v in static_columns.values()), default=0.0)
    max_val = max(max_temp, max_static, 1.0)  # Colour intensity relative

    num_temp_columns = len(temp_columns.keys())

    # Define figures and axes
    fig, axes = plt.subplots(
        nrows=num_temp_columns,
        ncols=1,
        sharex=True,  # Makes the plots share the same x axos
        figsize=(12, 8),
        constrained_layout=True,
    )

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=-max_val, vmax=max_val)

    # Plot data historic and future
    for i, col in enumerate(temp_columns.keys()):  # TODO: Again, is the df sorted here?
        if col in hist_df.columns:
            y = hist_df[col].tolist()
            x = range(len(y))
            axes[i].plot(x, y)
        else:
            y = []

        if col in fut_df.columns:
            y_fut = fut_df[col].tolist()
            x_fut = range(len(y), len(y) + len(y_fut))
            axes[i].plot(x_fut, y_fut, color="orange")

        axes[i].set_title(f"Variable: {col}")
        axes[i].set_xlabel("Time steps")

        # Plot vertical bars at segment indices
        for lag, (startx, endx) in segment_indices[col].items():
            axes[i].axvline(x=endx, color="darkgray", linestyle="--", linewidth=1.5)

            # Colour background according to importance weighting
            value = temp_columns[col][lag]
            color = cmap(norm(value))

            axes[i].axvspan(startx, endx, facecolor=color, alpha=0.3, linewidth=0, zorder=0)

    plt.show()


# STILL TODO:
## Plot future segments
## Sort variables alphabetically (or sim.) so axes always appear in same order
## Plot static variables
## Is plot a good way to present this info anyway?
## Could do some counterfactual equidistant spacing?
