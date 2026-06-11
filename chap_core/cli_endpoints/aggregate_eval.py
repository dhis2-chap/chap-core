"""Aggregate an evaluation .nc file up to parent org units defined by a GeoJSON."""

from __future__ import annotations

import json
import logging
from pathlib import Path  # noqa: TC003 — used at runtime via cyclopts get_type_hints()
from typing import Annotated

from cyclopts import Parameter

logger = logging.getLogger(__name__)


def aggregate_eval_cmd(
    input_file: Annotated[
        Path,
        Parameter(help="Evaluation NetCDF file produced by `chap eval`."),
    ],
    geojson_file: Annotated[
        Path,
        Parameter(help="GeoJSON whose features carry a `properties.parent` field."),
    ],
    output_file: Annotated[
        Path,
        Parameter("--output-file", help="Path to write the aggregated .nc file."),
    ],
) -> None:
    """Aggregate an evaluation .nc up to the parent org units defined by a GeoJSON.

    Forecast samples and observed cases are summed across siblings sharing the
    same ``properties.parent`` in the GeoJSON. Locations present in the .nc
    but missing from the GeoJSON are dropped with a warning; parents with only
    partial child coverage are summed over whichever children are present.

    Examples:
        chap aggregate-eval evaluation.nc admin1.geojson --output-file evaluation_admin0.nc
    """
    import xarray as xr

    from chap_core.geometry import Polygons

    logger.info(f"Loading evaluation from {input_file}")
    ds = xr.open_dataset(input_file)

    logger.info(f"Loading parent mapping from {geojson_file}")
    parent_dict = Polygons.from_file(geojson_file).get_parent_dict()

    locations = [str(loc) for loc in ds.coords["location"].values.tolist()]
    kept_locations: list[str] = []
    kept_parents: list[str] = []
    missing: list[str] = []
    for loc in locations:
        if loc in parent_dict:
            kept_locations.append(loc)
            kept_parents.append(parent_dict[loc])
        else:
            missing.append(loc)

    if missing:
        logger.warning(
            "Dropping %d location(s) not present in GeoJSON: %s",
            len(missing),
            ", ".join(missing),
        )

    if not kept_locations:
        raise ValueError(f"No locations from {input_file} were found in {geojson_file}; nothing to aggregate.")

    sliced = ds.sel(location=kept_locations)
    sliced = sliced.assign_coords(parent=("location", kept_parents))
    # ``min_count=1`` so a (time, horizon) cell whose children are *all* NaN aggregates to NaN
    # rather than 0. The default ``skipna=True`` sum returns 0 for an all-NaN group, which
    # fabricates spurious 0 forecasts at the ragged edges of a rolling-origin backtest (the
    # boundary target months only have some horizons populated). Those fake zeros then read as
    # real 0-forecasts in plots and corrupt downstream metrics. With min_count=1 a cell still
    # sums whichever children are present (preserving partial-coverage behaviour) but collapses
    # to NaN only when none are.
    aggregated = sliced.groupby("parent").sum(min_count=1)
    aggregated = aggregated.rename({"parent": "location"})

    aggregated.attrs.update(ds.attrs)
    aggregated.attrs["org_units"] = json.dumps(sorted(set(kept_parents)))
    aggregated.attrs["aggregation"] = f"aggregated_to_parent_from={input_file.name} via {geojson_file.name}"

    logger.info(f"Writing aggregated evaluation to {output_file}")
    aggregated.to_netcdf(output_file)
    ds.close()


def register_commands(app):
    app.command(name="aggregate-eval")(aggregate_eval_cmd)
