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

Endpoints are not implemented yet; each test is marked ``xfail(strict=True)``
so it passes today as an expected failure and starts failing loudly once the
implementation lands, prompting removal of the marker.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from chap_core.rest_api.app import app


client = TestClient(app)


CLIM_548 = "CLIM-548: faceting REST endpoints not implemented yet"


# @pytest.mark.xfail(strict=True, reason=CLIM_548)
def test_facet_coords_endpoint_returns_dict_per_dimension(override_session):
    response = client.get("/v1/visualization/backtest-plots/evaluation_plot/1/facet-coords")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, dict)
    assert "horizon_periods" in payload
    assert isinstance(payload["horizon_periods"], list)
    # EvaluationPlot declares ["location", "split_period"] as facet_dimensions.
    for dim in ("location", "split_period"):
        assert dim in payload, payload
        assert isinstance(payload[dim], list)
        assert len(payload[dim]) > 0


# @pytest.mark.xfail(strict=True, reason=CLIM_548)
def test_predicted_vs_actual_facet_coords_return_horizon_distance(override_session):
    response = client.get("/v1/visualization/backtest-plots/predicted_vs_actual/1/facet-coords")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, dict)
    assert "horizon_distance" in payload
    assert isinstance(payload["horizon_distance"], list)
    assert len(payload["horizon_distance"]) > 0
    assert "horizon_periods" in payload
    assert isinstance(payload["horizon_periods"], list)


# @pytest.mark.xfail(strict=True, reason=CLIM_548)
def test_subplot_endpoint_returns_vega_spec_for_coords(override_session):
    coords_resp = client.get("/v1/visualization/backtest-plots/evaluation_plot/1/facet-coords")
    assert coords_resp.status_code == 200, coords_resp.text
    payload = coords_resp.json()
    coords = {dim: values for dim, values in payload.items() if dim != "horizon_periods"}
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


# @pytest.mark.xfail(strict=True, reason=CLIM_548)
def test_subplots_endpoint_returns_one_entry_per_coord_combination(override_session):
    coords_resp = client.get("/v1/visualization/backtest-plots/evaluation_plot/1/facet-coords")
    payload = coords_resp.json()
    coords = {dim: values for dim, values in payload.items() if dim != "horizon_periods"}
    expected_count = 1
    for values in coords.values():
        expected_count *= len(values)
    assert "horizon_periods" in payload
    assert isinstance(payload["horizon_periods"], list)

    response = client.get("/v1/visualization/backtest-plots/evaluation_plot/1/subplots")
    assert response.status_code == 200, response.text
    entries = response.json()
    assert isinstance(entries, list)
    assert len(entries) == expected_count
    for entry in entries:
        assert "key" in entry
        assert "spec" in entry
        assert isinstance(entry["spec"], dict)
