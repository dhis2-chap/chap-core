"""
Script for plotting generated importance weighting with LIME
"""

import matplotlib.colors as mcolors
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


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
    max_val = max(max_temp, max_static, 1.0)

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=-max_val, vmax=max_val)

    sorted_vars = sorted(temp_columns.keys())
    has_static = bool(static_columns)
    n_rows = len(sorted_vars) + (1 if has_static else 0)

    fig = plt.figure(figsize=(14, max(4, n_rows * 2)), constrained_layout=True)
    gs = GridSpec(n_rows, 2, figure=fig, width_ratios=[85, 15])

    first_ax_hist = None
    for i, col in enumerate(sorted_vars):
        ax_hist = fig.add_subplot(gs[i, 0], sharex=first_ax_hist)
        ax_fut = fig.add_subplot(gs[i, 1])
        if first_ax_hist is None:
            first_ax_hist = ax_hist

        # Historical
        hist_indices = segment_indices.get(col)
        if col in hist_df.columns and hist_indices is not None:
            y = hist_df[col].tolist()
            ax_hist.plot(range(len(y)), y)
            for lag, (startx, endx) in hist_indices.items():
                ax_hist.axvline(x=endx, color="darkgray", linestyle="--", linewidth=1.0)
                value = temp_columns[col].get(lag, 0.0)
                ax_hist.axvspan(startx, endx, facecolor=cmap(norm(value)), alpha=0.3, zorder=0)
        else:
            ax_hist.axis("off")

        ax_hist.set_title(col, fontsize=9)
        ax_hist.tick_params(labelsize=7)

        # Future: single bar (horizon=1) or line + axvspan coloring (horizon>=2)
        fut_coeffs = {abs(lag): val for lag, val in temp_columns[col].items() if lag < 0}
        if col in fut_df.columns and fut_coeffs:
            y_fut = fut_df[col].tolist()
            if len(y_fut) == 1:
                color = cmap(norm(fut_coeffs.get(1, 0.0)))
                ax_fut.bar(0, y_fut[0], color=color, alpha=0.6, edgecolor="none")
            else:
                ax_fut.plot(range(len(y_fut)), y_fut)
                for j in range(len(y_fut)):
                    color = cmap(norm(fut_coeffs.get(j + 1, 0.0)))
                    ax_fut.axvspan(j - 0.5, j + 0.5, facecolor=color, alpha=0.3, zorder=0)
                for j in range(len(y_fut) - 1):
                    ax_fut.axvline(x=j + 0.5, color="darkgray", linestyle="--", linewidth=1.0)
            ax_fut.set_xticks(range(len(y_fut)))
            ax_fut.set_xticklabels([f"+{k + 1}" for k in range(len(y_fut))], fontsize=7)
            ax_fut.set_title("future", fontsize=8)
            ax_fut.tick_params(labelsize=7)
        else:
            ax_fut.axis("off")

    # Static features: one colored box per feature in the bottom row
    if has_static:
        ax_static = fig.add_subplot(gs[n_rows - 1, :])
        names = sorted(static_columns.keys())
        colors = [cmap(norm(static_columns[n])) for n in names]
        ax_static.bar(range(len(names)), [1] * len(names), color=colors, alpha=0.6, edgecolor="none")
        ax_static.set_xticks(range(len(names)))
        ax_static.set_xticklabels(names, fontsize=9)
        ax_static.set_yticks([])
        ax_static.set_title("Static features", fontsize=9)

    plt.show()
