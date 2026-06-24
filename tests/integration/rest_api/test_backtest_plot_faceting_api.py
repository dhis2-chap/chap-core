"""REST smoke tests for the CLIM-548 backtest plot faceting API.

Mirrors the business-logic smoke tests in
``tests/evaluation/test_backtest_plot_faceting.py``: documents the expected
HTTP surface for facet-coordinate discovery and subplot retrieval.

Expected endpoints (relative to ``/v1/visualization/backtest-plots``):

- ``GET  /{plot}/{backtest_id}/facet-coords``
  -> ``{"<dim>": [<value>, ...], ...}`` for every dim in ``facet_dimensions``.
- ``POST /{plot}/{backtest_id}/subplot``
  body: ``{"<dim>": <value>, ...}`` (one value per facet dimension)
  -> a single vega spec for that coordinate slice.
- ``GET  /{plot}/{backtest_id}/subplots``
  -> list of ``{"key": ..., "spec": {...}}`` pairs covering the full
  Cartesian product of facet coordinates.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from chap_core.rest_api.app import app


client = TestClient(app)


def test_facet_coords_endpoint_returns_dict_per_dimension(override_session):
    response = client.get("/v1/visualization/backtest-plots/evaluation_plot/1/facet-coords")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, dict)
    # EvaluationPlot declares ["location", "split_period"] as facet_dimensions.
    for dim in ("location", "split_period"):
        assert dim in payload, payload
        assert isinstance(payload[dim], list)
        assert len(payload[dim]) > 0


def test_predicted_vs_actual_facet_coords_return_horizon_distance(override_session):
    response = client.get("/v1/visualization/backtest-plots/predicted_vs_actual/1/facet-coords")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, dict)
    assert "horizon_distance" in payload
    assert isinstance(payload["horizon_distance"], list)
    assert len(payload["horizon_distance"]) > 0


def test_subplot_endpoint_returns_vega_spec_for_coords(override_session):
    coords_resp = client.get("/v1/visualization/backtest-plots/evaluation_plot/1/facet-coords")
    assert coords_resp.status_code == 200, coords_resp.text
    payload = coords_resp.json()
    coords = {dim: values for dim, values in payload.items() if dim != "horizon_distance"}
    body = {dim: coords[dim][0] for dim in coords}

    response = client.post(
        "/v1/visualization/backtest-plots/evaluation_plot/1/subplot",
        json=body,
    )
    assert response.status_code == 200, response.text
    spec = response.json()
    assert isinstance(spec, dict)
    # Vega specs always declare a schema URL.
    assert "$schema" in spec


def test_subplot_endpoint_uses_container_width(override_session):
    """The frontend embeds each subplot in a flexible-width container, so the API
    fills width via 'container' (height is left at the plot's fixed value)."""
    coords_resp = client.get("/v1/visualization/backtest-plots/evaluation_plot/1/facet-coords")
    payload = coords_resp.json()
    body = {dim: values[0] for dim, values in payload.items()}

    response = client.post("/v1/visualization/backtest-plots/evaluation_plot/1/subplot", json=body)
    assert response.status_code == 200, response.text
    spec = response.json()
    width_signal = next(s for s in spec["signals"] if s["name"] == "width")
    assert "containerSize" in width_signal["init"]


def test_subplot_endpoint_uses_container_width_for_vconcat_plot(override_session):
    """horizon_location_grid renders composite (vconcat) cells whose rows share one
    width; the API replaces the hoisted top-level width with a containerSize signal
    so the whole stack fills the frontend container."""
    coords_resp = client.get("/v1/visualization/backtest-plots/horizon_location_grid/1/facet-coords")
    assert coords_resp.status_code == 200, coords_resp.text
    payload = coords_resp.json()
    body = {dim: values[0] for dim, values in payload.items()}

    response = client.post("/v1/visualization/backtest-plots/horizon_location_grid/1/subplot", json=body)
    assert response.status_code == 200, response.text
    spec = response.json()
    assert "width" not in spec
    assert spec["autosize"]["type"] == "fit-x"
    width_signal = next(s for s in spec["signals"] if s["name"] == "width")
    assert "containerSize" in width_signal["init"]


def test_catalogue_lists_split_sample_bias_plots(override_session):
    response = client.get("/v1/visualization/backtest-plots/")
    assert response.status_code == 200, response.text
    plots = {entry["id"]: entry for entry in response.json()}
    assert "ratio_of_samples_above_truth" not in plots
    for plot_id in ("sample_bias_by_horizon", "sample_bias_by_time_period"):
        assert plot_id in plots
        assert plots[plot_id]["displayName"]
        assert plots[plot_id]["description"]


@pytest.mark.parametrize("plot_id", ["sample_bias_by_horizon", "sample_bias_by_time_period"])
def test_sample_bias_facet_coords_are_empty(plot_id, override_session):
    response = client.get(f"/v1/visualization/backtest-plots/{plot_id}/1/facet-coords")
    assert response.status_code == 200, response.text
    assert response.json() == {}


@pytest.mark.parametrize("plot_id", ["sample_bias_by_horizon", "sample_bias_by_time_period"])
def test_sample_bias_subplot_returns_single_view_container_spec(plot_id, override_session):
    """Each split plot must return a single-view spec sized to fill the container width."""
    response = client.post(f"/v1/visualization/backtest-plots/{plot_id}/1/subplot", json={})
    assert response.status_code == 200, response.text
    spec = response.json()
    assert "layout" not in spec
    assert spec["autosize"]["type"] == "fit-x"
    assert isinstance(spec["height"], (int, float))


def test_subplots_endpoint_returns_one_entry_per_coord_combination(override_session):
    coords_resp = client.get("/v1/visualization/backtest-plots/evaluation_plot/1/facet-coords")
    payload = coords_resp.json()
    coords = {dim: values for dim, values in payload.items() if dim != "horizon_distance"}
    expected_count = 1
    for values in coords.values():
        expected_count *= len(values)

    response = client.get("/v1/visualization/backtest-plots/evaluation_plot/1/subplots")
    assert response.status_code == 200, response.text
    entries = response.json()
    assert isinstance(entries, list)
    assert len(entries) == expected_count
    for entry in entries:
        assert "key" in entry
        assert "spec" in entry
        assert isinstance(entry["spec"], dict)
