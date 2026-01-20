"""
Utility functions for backtest plotting.

This module provides utility functions used by backtest visualizations.
The actual plot implementations are in chap_core.assessment.backtest_plots.
"""

import pandas as pd
import altair as alt
import textwrap

alt.data_transformers.enable("vegafusion")


def title_chart(text: str, width: int = 600, font_size: int = 24, pad: int = 10):
    """Return an Altair chart that just displays a title."""
    return (
        alt.Chart(pd.DataFrame({"x": [0], "y": [0]}))
        .mark_text(
            text=text,
            fontSize=font_size,
            fontWeight="bold",
            align="center",
            baseline="top",
        )
        .properties(width=width, height=font_size + pad)
    )


def text_chart(text, line_length=80, font_size=12, align="left", pad_bottom=50):
    import altair as alt
    import pandas as pd

    lines = textwrap.wrap(text, width=line_length)
    df = pd.DataFrame({"line": lines, "y": range(len(lines))})

    line_spacing = font_size + 2
    total_height = len(lines) * line_spacing + pad_bottom

    chart = (
        alt.Chart(df)
        .mark_text(align=align, baseline="top", fontSize=font_size)
        .encode(text="line", y=alt.Y("y:O", axis=None))
        .properties(height=total_height)
    )
    return chart


def clean_time(period):
    """Convert period to ISO date format for Altair/vegafusion compatibility."""
    if len(period) == 6:
        # YYYYMM format -> YYYY-MM-01 (add day for full date)
        return f"{period[:4]}-{period[4:]}-01"
    elif len(period) == 7 and period[4] == "-":
        # YYYY-MM format -> YYYY-MM-01 (add day for full date)
        return f"{period}-01"
    else:
        return period
