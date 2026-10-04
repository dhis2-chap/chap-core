"""Choropleth maps over a real basemap, the way chapkit's console draws them (MapLibre and OpenFreeMap)."""

from __future__ import annotations

import base64
import copy
import gzip
import html
import json
import math
from typing import Any

MAPLIBRE = "https://unpkg.com/maplibre-gl@5.9.0/dist/maplibre-gl"
# Free, no-API-key vector basemap, the same one chapkit and the open climate service use.
BASEMAP_STYLE = "https://tiles.openfreemap.org/styles/positron"
RAMP = ["#FEF0D9", "#FDCC8A", "#FC8D59", "#E34A33", "#B30000"]
NO_DATA_COLOR = "#C9CED6"
# Decimals kept in coordinates: five is about a metre, finer than any region border needs.
COORDINATE_DECIMALS = 5


def feature_location(feature: dict) -> str | None:
    """The location a GeoJSON feature stands for: its id, or properties.id."""
    location = feature.get("id", (feature.get("properties") or {}).get("id"))
    return None if location is None else str(location)


def bounding_box(geojson: dict) -> list[float] | None:
    """[west, south, east, north] of every coordinate in the features."""
    xs: list[float] = []
    ys: list[float] = []

    def walk(coords: Any) -> None:
        if coords and isinstance(coords[0], int | float):
            xs.append(coords[0])
            ys.append(coords[1])
        else:
            for item in coords:
                walk(item)

    for feature in geojson.get("features", []):
        if feature.get("geometry"):
            walk(feature["geometry"]["coordinates"])
    return [min(xs), min(ys), max(xs), max(ys)] if xs else None


def _rounded(coordinates: Any) -> Any:
    if isinstance(coordinates, float):
        return round(coordinates, COORDINATE_DECIMALS)
    return [_rounded(c) for c in coordinates] if isinstance(coordinates, list) else coordinates


def _json_safe(value: Any) -> Any:
    """The value with NaN and infinities as null: the browser's JSON parser refuses them."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return [_json_safe(v) for v in value] if isinstance(value, list) else value


def pack_features(geojson: dict) -> str:
    """The features as gzipped, base64-encoded GeoJSON with rounded coordinates.

    Chrome refuses a data: URL over 2 MB, and a province map can hold more than that in raw GeoJSON;
    packed it is a fraction of the size. The map page unpacks it with DecompressionStream.
    """
    packed = copy.deepcopy(geojson)
    for feature in packed.get("features", []):
        if feature.get("geometry"):
            feature["geometry"]["coordinates"] = _rounded(feature["geometry"]["coordinates"])
    text = json.dumps(_json_safe(packed), separators=(",", ":"), allow_nan=False)
    # mtime=0: without it gzip stamps the current time, and the same map would differ from second to second.
    return base64.b64encode(gzip.compress(text.encode(), mtime=0)).decode()


def choropleth_html(geojson: dict, values: dict[str, float], legend: str, height: int = 480) -> str:
    """A self-contained page with the regions coloured by value over a zoomable basemap."""
    features = copy.deepcopy(geojson)
    for feature in features.get("features", []):
        properties = feature.setdefault("properties", {}) or {}
        feature["properties"] = properties
        location = feature_location(feature)
        properties["label"] = properties.get("name") or location or ""
        if location in values and math.isfinite(values[location]):
            properties["value"] = values[location]
    present = [v for v in values.values() if v == v]
    low, high = (min(present), max(present)) if present else (0.0, 1.0)
    if high == low:
        high = low + 1
    stops = [low + (high - low) * i / (len(RAMP) - 1) for i in range(len(RAMP))]
    fill: list[Any] = ["interpolate", ["linear"], ["get", "value"]]
    for stop, color in zip(stops, RAMP, strict=True):
        fill += [stop, color]
    config = {
        "style": BASEMAP_STYLE,
        "features": pack_features(features),
        "bounds": bounding_box(features),
        "fill": ["case", ["has", "value"], fill, NO_DATA_COLOR],
    }
    gradient = ", ".join(RAMP)
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="{MAPLIBRE}.css">
<script src="{MAPLIBRE}.js"></script>
<style>
html, body {{ margin: 0; height: 100%; font-family: "IBM Plex Sans", system-ui, sans-serif; }}
#map {{ position: absolute; inset: 0; }}
.legend {{ position: absolute; left: 12px; bottom: 12px; background: #fff; padding: 8px 10px; border-radius: 6px;
  box-shadow: 0 1px 3px rgba(0,0,0,.15); font-size: 12px; color: #15191E; min-width: 180px; }}
.ramp {{ height: 10px; border-radius: 2px; background: linear-gradient(to right, {gradient}); margin: 6px 0 2px; }}
.ends {{ display: flex; justify-content: space-between; color: #5B6470; }}
.maplibregl-popup-content {{ font-size: 12px; padding: 6px 10px; }}
</style></head>
<body style="height:{height}px">
<div id="map"></div>
<div class="legend">{html.escape(legend)}<div class="ramp"></div>
<div class="ends"><span>{low:.3g}</span><span>{high:.3g}</span></div></div>
<script>
const config = {json.dumps(config).replace("<", "\\u003c")};
const map = new maplibregl.Map({{
  container: "map", style: config.style, attributionControl: {{ compact: true }},
  // Keep the drawn frame so the map can be captured, e.g. saved as an image or screenshotted.
  canvasContextAttributes: {{ preserveDrawingBuffer: true }},
}});
map.addControl(new maplibregl.NavigationControl({{ showCompass: false }}), "top-right");
const fit = () => config.bounds && map.fitBounds(
  [[config.bounds[0], config.bounds[1]], [config.bounds[2], config.bounds[3]]], {{ padding: 40, duration: 0 }});
fit();
// The map can start hidden (e.g. in a closed tab) with no size; redraw and refit once it becomes visible.
let fitted = false;
new ResizeObserver(() => {{
  map.resize();
  if (!fitted && map.getContainer().clientWidth > 0) {{ fitted = true; fit(); }}
}}).observe(document.getElementById("map"));
// The regions come gzipped (see pack_features), so a detailed map still fits in the page's data: URL.
const features = new Response(
  new Blob([Uint8Array.from(atob(config.features), (c) => c.charCodeAt(0))]).stream()
    .pipeThrough(new DecompressionStream("gzip"))).json();
map.on("load", async () => {{
  map.addSource("locations", {{ type: "geojson", data: await features }});
  map.addLayer({{ id: "fill", type: "fill", source: "locations",
    paint: {{ "fill-color": config.fill, "fill-opacity": 0.78 }} }});
  map.addLayer({{ id: "outline", type: "line", source: "locations",
    paint: {{ "line-color": "#ffffff", "line-width": 1 }} }});
  const popup = new maplibregl.Popup({{ closeButton: false, closeOnClick: false }});
  map.on("mousemove", "fill", (e) => {{
    const p = e.features[0].properties;
    const value = p.value === undefined ? "no data" : Number(p.value).toFixed(3);
    map.getCanvas().style.cursor = "pointer";
    const box = document.createElement("div");
    const name = document.createElement("b");
    name.textContent = p.label;
    box.append(name, document.createElement("br"), document.createTextNode(value));
    popup.setLngLat(e.lngLat).setDOMContent(box).addTo(map);
  }});
  map.on("mouseleave", "fill", () => {{ map.getCanvas().style.cursor = ""; popup.remove(); }});
}});
</script></body></html>"""


def choropleth_url(geojson: dict, values: dict[str, float], legend: str, height: int = 480) -> str:
    """The map page as a data: URL, for `st.iframe`.

    Embedded as raw HTML the page shares the app's origin only partly, and MapLibre's web worker
    never answers; as a data: URL the page and its worker share one origin and the map loads.
    """
    page = choropleth_html(geojson, values, legend, height)
    return "data:text/html;base64," + base64.b64encode(page.encode()).decode()
