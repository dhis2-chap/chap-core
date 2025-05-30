import json
import logging
from typing import Tuple, Dict

import pooch
import pycountry
from pydantic_geojson import FeatureModel, FeatureCollectionModel
from unidecode import unidecode
from .api_types import (
    FeatureCollectionModel as DFeatureCollectionModel,
    FeatureModel as DFeatureModel,
)

logger = logging.getLogger(__name__)


class PFeatureModel(FeatureModel):
    properties: dict


class PFeatureCollectionModel(FeatureCollectionModel):
    features: list[PFeatureModel]


data_path = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{country_code}_{level}.json.zip"

# country_codes= {'vietnam': 'VNM', 'laos': 'LAO', 'cambodia': 'KHM', 'thailand': 'THA', 'myanmar': 'MMR', 'brazil': 'BRA', 'colombia': 'COL', 'peru': 'PER', 'ecuador': 'ECU', 'bolivia': 'BOL', 'paraguay': 'PRY'}


country_names = [
    "brazil",
    "mexico",
    "el salvador",
    "paraguay",
    "peru",
    "colombia",
    "ecuador",
    "nicaragua",
    "panama",
    "argentina",
    "indonesia",
    "philippines",
    "thailand",
    "vietnam",
    "laos",
    "malaysia",
    "cambodia",
    "singapore",
]
country_codes_l = [
    "BRA",
    "MEX",
    "SLV",
    "PRY",
    "PER",
    "COL",
    "ECU",
    "NIC",
    "PAN",
    "ARG",
    "IDN",
    "PHL",
    "THA",
    "VNM",
    "LAO",
    "MYS",
    "KHM",
    "SGP",
]
country_codes = dict(zip(country_names, country_codes_l))


def normalize_name(name: str) -> str:
    return unidecode(name.replace(" ", "").lower())


def add_id(feature, admin_level=1, lookup_dict=None):
    id = feature.properties[f"NAME_{admin_level}"]
    if lookup_dict:
        id = lookup_dict[normalize_name(id)]
    return DFeatureModel(**feature.model_dump(), id=id)


def get_area_polygons(country: str, regions: list[str] = None, admin_level: int = 1) -> FeatureCollectionModel:
    """
    Get the polygons for the specified regions in the specified country (only ADMIN1 supported)
    Returns only the regions that are found in the data
    Name is put in the id field of the feature

    Parameters
    ----------
    country : str
        The country name
    regions : list[str], optional
        Filtering which regions to get the polygons for based on region name
    admin_level : int
        Which administrative level to get polygons for (0 for country, 1 for admin-1, etc)

    Returns
    -------
    FeatureCollectionModel
        The polygons for the specified regions

    """

    # TODO: switch feature_dict to use ID_{admin_level} instead of name (there can be duplicate names)

    data = get_country_data(country, admin_level=admin_level)
    feature_dict = {normalize_name(feature.properties[f"NAME_{admin_level}"]): feature for feature in data.features}
    for f in feature_dict.values():
        print(f.properties)
    logger.info(f"Polygon data available for regions: {list(feature_dict.keys())}")
    if regions:
        logger.info(f"Filtering to requested regions: {[normalize_name(region) for region in regions]}")
        normalized_to_original = {normalize_name(region): region for region in regions}
        features = [
            add_id(feature_dict[normalize_name(region)], admin_level, normalized_to_original)
            for region in regions
            if normalize_name(region) in feature_dict
        ]
    else:
        features = [add_id(feature, admin_level) for feature in feature_dict.values()]

    return DFeatureCollectionModel(
        type="FeatureCollection",
        features=features,
    )


def get_country_data_file(country: str, level=1):
    real_name = country.capitalize()
    print(real_name)
    country_code = pycountry.countries.get(name=real_name).alpha_3
    # country_code = country_codes[country.lower()]
    return get_data_file(country_code, level)


def get_data_file(country_code: str, level=1):
    data_url = data_path.format(country_code=country_code, level=level)
    return pooch.retrieve(data_url, None)


def get_country_data(country, admin_level) -> PFeatureCollectionModel:
    zip_filaname = get_country_data_file(country, admin_level)
    # read zipfile
    import zipfile

    with zipfile.ZipFile(zip_filaname) as z:
        filaname = z.namelist()[0]
        with z.open(filaname) as f:
            return PFeatureCollectionModel.model_validate_json(f.read())

    # return DFeatureCollectionModel.model_validate_json(open(filaname).read())


def get_all_data():
    return ((country, get_country_data(country)) for country in country_codes.keys())


class Polygons:
    def __init__(self, polygons):
        self._polygons = polygons

    def __eq__(self, other):
        return self._polygons == other._polygons

    def __len__(self):
        return len(self._polygons.features)

    def __iter__(self):
        for feat in self._polygons.features:
            yield feat

    @property
    def __geo_interface__(self):
        return self.to_geojson()

    @property
    def data(self) -> FeatureCollectionModel:
        return self._polygons

    @property
    def bbox(self):
        from .geoutils import feature_bbox

        boxes = (feature_bbox(feat) for feat in self)
        xmins, ymins, xmaxs, ymaxs = zip(*boxes)
        bbox = min(xmins), min(ymins), max(xmaxs), max(ymaxs)
        return bbox

    @classmethod
    def from_file(cls, filename, id_property="id"):
        return cls.from_geojson(json.load(open(filename)), id_property=id_property)

    def to_file(self, filename):
        json.dump(self.to_geojson(), open(filename, "w"))

    def id_to_name_tuple_dict(self) -> Dict[str, Tuple[str]]:
        """Returns a dictionary with the id as key and a tuple with the name and parent as value"""
        lookup = {}
        for feat in self.feature_collection().features:
            if feat.properties is None:
                lookup[feat.id] = (feat.id, "-")
                continue
            lookup[feat.id] = (feat.properties.get("name", feat.id), feat.properties.get("parent", "-"))
        return lookup

    def get_parent_dict(self):
        return {
            feature.id: feature.properties.get("parent", "-") if feature.properties else "-"
            for feature in self._polygons.features
        }

    def get_predecessors_map(self, predecessors: list[str]):
        predecessors = set(predecessors)
        map = {}
        for feature in self._polygons.features:
            if feature.properties is None:
                raise ValueError(f"Feature {feature.id} has no properties")
            if "parentGraph" not in feature.properties:
                raise ValueError(f"Feature {feature.id} has no parentGraph property")
            possible_parents = [p for p in feature.properties["parentGraph"].split("/") if p in predecessors]
            if len(possible_parents) == 0:
                raise ValueError(f"Feature {feature.id} has no parent in predecessors")
            if len(possible_parents) > 1:
                raise ValueError(f"Feature {feature.id} has multiple parents in predecessors")
            map[feature.id] = possible_parents[0]
        return map

    def filter_locations(self, locations: list[str]):
        """
        Filter the polygons to only include the specified locations
        """
        filtered = [feature for feature in self._polygons.features if feature.id in locations]
        return Polygons(DFeatureCollectionModel(features=filtered))

    @classmethod
    def _add_ids(cls, features: DFeatureCollectionModel, id_property: str):
        for feature in features.features:
            try:
                feature.id = feature.id or unidecode(str(feature.properties[id_property]))
            except AttributeError:
                logging.error(f"{feature.properties[id_property]}")
                raise
        return features

    @classmethod
    def from_geojson(cls, geojson: dict, id_property: str = "id"):
        # TODO: check or skip non-Polygon features
        # ...

        # validate features one-by-one
        errors = 0
        features = []
        for feat in geojson["features"]:
            # skip invalid features
            try:
                feat = DFeatureModel.model_validate(feat)
                features.append(feat)
            except Exception:
                errors += 1

        # warn user about skipped features
        if errors:
            logger.warning(f"Skipped {errors} geojson features due to failed validation")

        # create pydantic feature collection and add ids
        features = DFeatureCollectionModel(features=features)
        features = cls._add_ids(features, id_property)

        # init class object
        return cls(features)

    def to_geojson(self):
        return self._polygons.model_dump()

    def feature_collection(self):
        return self._polygons


if __name__ == "__main__":
    base_filename = "/home/knut/Data/ch_data/geometry"
    for country, data in get_all_data():
        json.dump(data.model_dump(), open(f"{base_filename}/{country}.json", "w"))
