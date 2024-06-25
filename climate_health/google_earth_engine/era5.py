import ee
import dataclasses
import enum
import os
from dotenv import load_dotenv, find_dotenv

@dataclasses.dataclass
class Result:
    orgUntId: str
    dhis2Periode : str
    value : str

class AggregationPeriode(enum.Enum):
    MONTHLY = 'CMWF/ERA5_LAND/MONTHLY_AGGR'
    DAILY = 'ECMWF/ERA5_LAND/DAILY_AGGR'

band = ["temperature_2m"]
reducer = "mean"

monthlyDataset = {
    "datasetId" : "ECMWF/ERA5_LAND/MONTHLY_AGGR",
    "band" : band,
    "reducer" : reducer,
}

dailyDataset = {
    "datasetId" : "ECMWF/ERA5_LAND/DAILY_AGGR",
    "band" : band,
    "reducer" : reducer,
}

class GoogleEarthEngine:
    def __init__(self):
        self.intitilizeClient(self)

    def intitilizeClient(self):

        #Load environment variables
        load_dotenv(find_dotenv())

        return
        
        #read environment variables
        account = os.environ['GOOGLE_SERVICE_ACCOUNT_EMAIL']
        private_key = os.environ['GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY']

        if(not account):
           print("GOOGLE_SERVICE_ACCOUNT_EMAIL is not set, you need to set it in the environment variables")
        if(not private_key):
           print("GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY is not set, you need to set it in the environment variables")

        print(account, private_key)
        return

        credentials = ee.ServiceAccountCredentials(account, key_file=None, key_data=private_key)
        ee.Initialize(credentials)
        

    def fetch_data_climate_indicator(features: object, start_data: str, end_date: str):

        #if polygon
        start_data = '2019-01-01'
        end_date = '2019-01-02'

        collection = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').select(band).filterDate(start_data, end_date).mean()

        print(collection.getInfo())

        if(collection.size() == 0):
            raise Exception("No data found for the given period, collection returned 0 elements")

        if (reducer == "min" or reducer == "max"):
            # ReduceRegions with min/max reducer may fail if the features are smaller than the pixel area
            # https://stackoverflow.com/questions/59774022/reduce-regions-some-features-dont-contains-centroid-of-pixel-in-consecuence-ex
            
            # TODO Handle this
            return

        featureCollection = ee.FeatureCollection(features)

        #TODO find out, what is this doing?
        eeReducer = ee.Reducer[reducer]()





        