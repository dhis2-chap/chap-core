from typing import IO
from pathlib import Path
from libpysal.weights import Queen

import geopandas as gpd


def geojson_to_shape(geojson_filename: str, shape_filename: str | Path):
    gdf = gpd.read_file(geojson_filename)
    gdf.to_file(shape_filename)


def geojson_to_graph(geojson_filename: str | IO, graph_filename: str | Path):  # , graph_filename: str|Path):
    NeighbourGraph.from_geojson_file(geojson_filename).to_graph_file(graph_filename)


class LocationMapping:
    def __init__(self, ordered_locations):
        self._location_map = {i + 1: location for i, location in enumerate(ordered_locations)}
        self._reverse_map = {v: k for k, v in self._location_map.items()}

    def name_to_index(self, name):
        assert name in self._reverse_map, f"Name {name} not found in location map {[self._reverse_map.keys()]}"
        return self._reverse_map[name]

    def index_to_name(self, item):
        return self._location_map[item]


class NeighbourGraph:
    @classmethod
    def from_geojson_file(cls, geo_json_file: IO):
        regions = gpd.read_file(geo_json_file)
        print(regions)
        graph = Queen.from_dataframe(regions)
        return cls(regions, graph)

    def __init__(self, regions, graph):
        self._regions = regions
        self._graph = graph
        self.location_map = LocationMapping(self._regions["id"])

    def __str__(self):
        return str(self._graph.neighbors.items())

    def to_graph_file(self, graph_filename: str | Path):
        with open(graph_filename, "w") as f:
            f.write(f"{len(self._graph.neighbors)}\n")
            for from_id, neighbours in self._graph.neighbors.items():
                L = len(neighbours)
                vals = [from_id + 1, L] + [n + 1 for n in neighbours]
                f.write(" ".join(map(str, vals)) + "\n")
        return True
