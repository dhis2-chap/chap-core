import json

import pooch
import pycountry
from pydantic_geojson import FeatureModel, FeatureCollectionModel
from unidecode import unidecode
from .api_types import (
    FeatureCollectionModel as DFeatureCollectionModel,
    FeatureModel as DFeatureModel,
)


class PFeatureModel(FeatureModel):
    properties: dict


class PFeatureCollectionModel(FeatureCollectionModel):
    features: list[PFeatureModel]


data_path = (
    "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{country_code}_1.json.zip"
)

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


def add_id(feature):
    id = feature.properties["NAME_1"]
    return DFeatureModel(**feature.dict(), id=id)
    return feature


def get_area_polygons(country: str, regions: list[str]) -> FeatureCollectionModel:
    """
    Get the polygons for the specified regions in the specified country (only ADMIN1 supported)
    Returns only the regions that are found in the data
    Name is put in the id field of the feature

    Parameters
    ----------
    country : str
        The country name
    regions : list[str]
        The regions to get the polygons for

    Returns
    -------
    FeatureCollectionModel
        The polygons for the specified regions

    """
    data = get_country_data(country)
    feature_dict = {
        normalize_name(feature.properties["NAME_1"]): feature
        for feature in data.features
    }
    return DFeatureCollectionModel(
        type="FeatureCollection",
        features=[
            add_id(feature_dict[normalize_name(region)])
            for region in regions
            if normalize_name(region) in feature_dict
        ],
    )


def get_country_data_file(country: str):
    real_name = country.capitalize()
    print(real_name)
    country_code = pycountry.countries.get(name=real_name).alpha_3
    # country_code = country_codes[country.lower()]
    return get_data_file(country_code)


def get_data_file(country_code: str):
    data_url = data_path.format(country_code=country_code)
    return pooch.retrieve(data_url, None)


def get_country_data(country) -> PFeatureCollectionModel:
    zip_filaname = get_country_data_file(country)
    # read zipfile
    import zipfile

    with zipfile.ZipFile(zip_filaname) as z:
        filaname = z.namelist()[0]
        with z.open(filaname) as f:
            return PFeatureCollectionModel.model_validate_json(f.read())

    # return DFeatureCollectionModel.model_validate_json(open(filaname).read())


def get_all_data():
    return ((country, get_country_data(country)) for country in country_codes.keys())


if __name__ == "__main__":
    base_filename = "/home/knut/Data/ch_data/geometry"
    for country, data in get_all_data():
        json.dump(data.model_dump(), open(f"{base_filename}/{country}.json", "w"))
