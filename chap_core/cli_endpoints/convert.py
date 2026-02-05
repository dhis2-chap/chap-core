"""Convert commands for CHAP CLI."""

import json
import logging
from pathlib import Path
from typing import Annotated

import pandas as pd
from cyclopts import Parameter

logger = logging.getLogger(__name__)


def convert_request(
    request_json: Annotated[
        Path,
        Parameter(help="Path to a create-backtest-with-data JSON request file"),
    ],
    output_prefix: Annotated[
        Path,
        Parameter(help="Prefix for output files (creates PREFIX.csv and PREFIX.geojson)"),
    ],
):
    """Convert a create-backtest-with-data JSON request to CSV and GeoJSON files.

    Takes a JSON payload from the DHIS2/Modeling App and produces:
    1. A CHAP-compatible CSV file with time_period, location, and feature columns
    2. A GeoJSON file with region boundaries

    Examples:
        chap convert-request ./request.json ./output
    """
    with open(request_json) as f:
        data = json.load(f)

    provided_data = data["providedData"]
    df = pd.DataFrame(provided_data)

    df = df.rename(columns={"orgUnit": "location", "period": "time_period"})

    pivoted = df.pivot_table(
        index=["location", "time_period"],
        columns="featureName",
        values="value",
        aggfunc="first",
    ).reset_index()

    pivoted.columns.name = None

    csv_path = Path(f"{output_prefix}.csv")
    pivoted.to_csv(csv_path, index=False)
    print(f"Created: {csv_path}")

    geojson_path = Path(f"{output_prefix}.geojson")
    with open(geojson_path, "w") as f:
        json.dump(data["geojson"], f, indent=2)
    print(f"Created: {geojson_path}")


def register_commands(app):
    """Register convert commands with the CLI app."""
    app.command(name="convert-request")(convert_request)
