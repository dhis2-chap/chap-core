from dotenv import find_dotenv, load_dotenv
from climate_health.api import read_zip_folder, train_on_prediction_data
from climate_health.google_earth_engine.gee_era5 import Era5LandGoogleEarthEngine

def initialize_gee_client():
    gee_client = Era5LandGoogleEarthEngine()
    return gee_client

def train_on_zip_file(file, model_name, model_path, control=None):
    
    gee_client = initialize_gee_client()
    
    print('train_on_zip_file')
    print('F', file)
    prediction_data = read_zip_folder(file.file)

    prediction_data.climate_data = gee_client.get_historical_era5_from_gee(file.file, prediction_data.health_data.period_range)
    

    return train_on_prediction_data(prediction_data, model_name=model_name, model_path=model_path, control=control)