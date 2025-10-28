import pytest
import altair as alt

from chap_core.plotting.dataset_plot import StandardizedFeaturePlot, DiseaseCasesMap
from chap_core.plotting.season_plot import SeasonCorrelationBarPlot


@pytest.fixture()
def default_transformer():
    import altair as alt

    alt.data_transformers.enable("default")
    yield


@pytest.mark.parametrize("plt_cls", [DiseaseCasesMap, StandardizedFeaturePlot, SeasonCorrelationBarPlot])
def test_standardized_feautre_plot(simulated_dataset, plt_cls, default_transformer):
    plotter = plt_cls.from_dataset_model(simulated_dataset)
    chart = plotter.plot()


@pytest.mark.skip(reason="Gives a lot of plots")
def test_dummy_geojson_boundaries(dummy_geojson, tmp_path):
    """Test to visualize the dummy GeoJSON boundaries using matplotlib."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MPLPolygon

    fig, ax = plt.subplots(figsize=(12, 10))

    colors = ["lightblue", "lightgreen", "lightcoral"]

    for idx, feature in enumerate(dummy_geojson["features"]):
        coords = feature["geometry"]["coordinates"][0]
        polygon = MPLPolygon(
            coords, facecolor=colors[idx], edgecolor="black", linewidth=2, label=feature["properties"]["name"]
        )
        ax.add_patch(polygon)

        # Add label in center of polygon
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        center_lon = sum(lons) / len(lons)
        center_lat = sum(lats) / len(lats)
        ax.text(
            center_lon,
            center_lat,
            feature["properties"]["id"],
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    # Set axis limits with some padding
    all_lons = []
    all_lats = []
    for feature in dummy_geojson["features"]:
        coords = feature["geometry"]["coordinates"][0]
        all_lons.extend([c[0] for c in coords])
        all_lats.extend([c[1] for c in coords])

    padding = 0.2
    ax.set_xlim(min(all_lons) - padding, max(all_lons) + padding)
    ax.set_ylim(min(all_lats) - padding, max(all_lats) + padding)

    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title("Dummy GeoJSON Boundaries", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_aspect("equal")

    # Save to file
    output_file = tmp_path / "dummy_geojson_boundaries.png"
    plt.savefig(str(output_file), dpi=150, bbox_inches="tight")
    print(f"\nMap saved to: {output_file}")

    # Also save to current directory for easy viewing
    plt.savefig("dummy_geojson_boundaries.png", dpi=150, bbox_inches="tight")
    print("Also saved to: dummy_geojson_boundaries.png")

    plt.close()
