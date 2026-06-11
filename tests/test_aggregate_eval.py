"""Tests for the `chap aggregate-eval` CLI command."""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest
import xarray as xr

from chap_core.cli_endpoints.aggregate_eval import aggregate_eval_cmd


def _build_eval_dataset(tmp_path):
    """Build a minimal evaluation .nc with locations a, b, c."""
    locations = ["a", "b", "c"]
    time_periods = ["2024-01", "2024-02"]
    horizons = [1, 2]
    samples = [0, 1]

    # Distinct integer values per (location, time, horizon, sample) so we can
    # check sums by adding the underlying arrays.
    rng = np.arange(len(locations) * len(time_periods) * len(horizons) * len(samples), dtype=float).reshape(
        len(locations), len(time_periods), len(horizons), len(samples)
    )

    observed = np.arange(len(locations) * len(time_periods), dtype=float).reshape(len(locations), len(time_periods))

    ds = xr.Dataset(
        data_vars={
            "forecast": (("location", "time_period", "horizon_distance", "sample"), rng),
            "observed": (("location", "time_period"), observed),
        },
        coords={
            "location": locations,
            "time_period": time_periods,
            "horizon_distance": horizons,
            "sample": samples,
        },
        attrs={
            "title": "test",
            "model_name": "test-model",
            "org_units": json.dumps(locations),
        },
    )
    nc_path = tmp_path / "eval.nc"
    ds.to_netcdf(nc_path)
    return nc_path, ds


def _write_geojson(path, features):
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": feat_id,
                "properties": {"parent": parent},
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
            }
            for feat_id, parent in features
        ],
    }
    path.write_text(json.dumps(geojson))


def test_aggregate_eval_sums_children(tmp_path):
    nc_path, original = _build_eval_dataset(tmp_path)
    geojson_path = tmp_path / "areas.geojson"
    _write_geojson(geojson_path, [("a", "P"), ("b", "P"), ("c", "Q")])
    out_path = tmp_path / "aggregated.nc"

    aggregate_eval_cmd(nc_path, geojson_path, out_path)

    assert out_path.exists()
    result = xr.open_dataset(out_path)
    try:
        assert sorted(result.coords["location"].values.tolist()) == ["P", "Q"]

        expected_p_forecast = (
            original["forecast"].sel(location="a").values + original["forecast"].sel(location="b").values
        )
        np.testing.assert_array_equal(result["forecast"].sel(location="P").values, expected_p_forecast)
        np.testing.assert_array_equal(
            result["forecast"].sel(location="Q").values,
            original["forecast"].sel(location="c").values,
        )

        expected_p_observed = (
            original["observed"].sel(location="a").values + original["observed"].sel(location="b").values
        )
        np.testing.assert_array_equal(result["observed"].sel(location="P").values, expected_p_observed)

        assert json.loads(result.attrs["org_units"]) == ["P", "Q"]
        assert result.attrs["model_name"] == "test-model"
        assert "aggregation" in result.attrs
    finally:
        result.close()


def test_aggregate_eval_drops_unknown_locations(tmp_path, caplog):
    nc_path, original = _build_eval_dataset(tmp_path)
    geojson_path = tmp_path / "areas.geojson"
    _write_geojson(geojson_path, [("a", "P"), ("b", "P")])  # c omitted
    out_path = tmp_path / "aggregated.nc"

    with caplog.at_level(logging.WARNING):
        aggregate_eval_cmd(nc_path, geojson_path, out_path)

    assert any("c" in record.message for record in caplog.records)

    result = xr.open_dataset(out_path)
    try:
        assert result.coords["location"].values.tolist() == ["P"]
        expected_p_forecast = (
            original["forecast"].sel(location="a").values + original["forecast"].sel(location="b").values
        )
        np.testing.assert_array_equal(result["forecast"].sel(location="P").values, expected_p_forecast)
    finally:
        result.close()


def test_aggregate_eval_errors_when_no_locations_match(tmp_path):
    nc_path, _ = _build_eval_dataset(tmp_path)
    geojson_path = tmp_path / "areas.geojson"
    _write_geojson(geojson_path, [("x", "P"), ("y", "Q")])
    out_path = tmp_path / "aggregated.nc"

    with pytest.raises(ValueError, match="nothing to aggregate"):
        aggregate_eval_cmd(nc_path, geojson_path, out_path)


def test_aggregate_eval_preserves_nan_for_all_nan_cells(tmp_path):
    """A (time, horizon) cell where every child is NaN must aggregate to NaN, not 0.

    Rolling-origin backtests leave the boundary (target month, horizon) cells unpopulated
    (NaN). A plain skipna sum would turn an all-NaN parent cell into 0, fabricating a spurious
    0 forecast; aggregation must keep it NaN. A cell with *some* children present should still
    sum the present ones.
    """
    _, original = _build_eval_dataset(tmp_path)
    # Make one (time, horizon) cell all-NaN for both children of parent P (a, b),
    # and another cell NaN for only one child (a) to check partial coverage still sums.
    ds = original.copy(deep=True)
    ds["forecast"].loc[dict(location=["a", "b"], time_period="2024-01", horizon_distance=1)] = np.nan
    ds["forecast"].loc[dict(location="a", time_period="2024-02", horizon_distance=1)] = np.nan
    nc_path = tmp_path / "eval_nan.nc"
    ds.to_netcdf(nc_path)

    geojson_path = tmp_path / "areas.geojson"
    _write_geojson(geojson_path, [("a", "P"), ("b", "P"), ("c", "Q")])
    out_path = tmp_path / "aggregated.nc"
    aggregate_eval_cmd(nc_path, geojson_path, out_path)

    result = xr.open_dataset(out_path)
    try:
        # all-NaN cell -> NaN (the bug would make this 0)
        all_nan_cell = result["forecast"].sel(location="P", time_period="2024-01", horizon_distance=1).values
        assert np.isnan(all_nan_cell).all()
        # partial cell -> sum of the present child only (b), NaN child (a) skipped
        partial = result["forecast"].sel(location="P", time_period="2024-02", horizon_distance=1).values
        expected_b = ds["forecast"].sel(location="b", time_period="2024-02", horizon_distance=1).values
        np.testing.assert_array_equal(partial, expected_b)
    finally:
        result.close()
