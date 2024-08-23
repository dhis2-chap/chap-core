import pooch

from pydantic_geojson import FeatureModel, FeatureCollectionModel


class DFeatureModel(FeatureModel):
    properties: dict

class DFeatureCollectionModel(FeatureCollectionModel):
    features: list[DFeatureModel]

data_path = 'https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{country_code}_1.json.zip'

#country_codes= {'vietnam': 'VNM', 'laos': 'LAO', 'cambodia': 'KHM', 'thailand': 'THA', 'myanmar': 'MMR', 'brazil': 'BRA', 'colombia': 'COL', 'peru': 'PER', 'ecuador': 'ECU', 'bolivia': 'BOL', 'paraguay': 'PRY'}

country_names =['brazil', 'mexico', 'el salvador', 'paraguay', 'peru', 'colombia', 'ecuador', 'nicaragua', 'panama', 'argentina', 'indonesia', 'philippines', 'thailand', 'vietnam', 'laos', 'malaysia', 'cambodia', 'singapore']
country_codes_l = ['BRA', 'MEX', 'SLV', 'PRY', 'PER', 'COL', 'ECU', 'NIC', 'PAN', 'ARG', 'IDN', 'PHL', 'THA', 'VNM', 'LAO', 'MYS', 'KHM', 'SGP']
country_codes = dict(zip(country_names, country_codes_l))

def get_country_data_file(country: str):
    country_code = country_codes[country.lower()]
    return get_data_file(country_code)

def get_data_file(country_code: str):
    data_url = data_path.format(country_code=country_code)
    return pooch.retrieve(data_url, None)

def get_country_data(country)->DFeatureCollectionModel:
    zip_filaname = get_country_data_file(country)
    # read zipfile
    import zipfile
    with zipfile.ZipFile(zip_filaname) as z:
        filaname = z.namelist()[0]
        with z.open(filaname) as f:
            return DFeatureCollectionModel.model_validate_json(f.read())

    #return DFeatureCollectionModel.model_validate_json(open(filaname).read())


def get_all_data():
    return {country: get_country_data(country) for country in country_codes.keys()}

if __name__ == '__main__':
    print(get_all_data())



