import datetime
import json
from typing import List
import ee
import dataclasses
import enum
import os
from array import array
import zipfile

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

@dataclasses.dataclass
class Periode:
    id: str
    startDate : datetime
    endDate : datetime

class GoogleEarthEngine:

    #def getInfo(self, instance):
    #    return instance.evaluate()

    def kelvinToCelsius(self, k):
        return k - 273.15

    def roundOneDecimal(self, v):
        return round(v * 10) / 10

    def valueParser(self, v):
        return self.roundOneDecimal(self.kelvinToCelsius(v))
    
    def __init__(self):
        self.initializeClient()

    def initializeClient(self):
        #read environment variables
        account = os.environ.get('GOOGLE_SERVICE_ACCOUNT_EMAIL')
        private_key = os.environ.get('GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY')

        if(not account):
            print("GOOGLE_SERVICE_ACCOUNT_EMAIL is not set, you need to set it in the environment variables to use Google Earth Engine")
        if(not private_key):
            print("GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY is not set, you need to set it in the environment variables to use Google Earth Engine")

        if(not account or not private_key):
            return
        
        try:
            credentials = ee.ServiceAccountCredentials(account, key_data=private_key)
            ee.Initialize(credentials)
            print("Google Earth Engine initialized, with account:", account)
        except ValueError as e:
            print("\nERROR:\n", e, "\n")

    def dataParser(self, data):
        parsed_data = [ 
                { **f['properties'], 
                 'period': f['properties']['period'], 
                 'value': self.valueParser(f['properties']['value']) 
            if self.valueParser else f['properties']['value'] } for f in data 
            ]
        return parsed_data

    '''
        This method, takes in the ZIP-file from the post-file path, and fetches the data from Google Earth Engine
        Data is beeing 
    '''
    def fetch_data_climate_indicator(self, zip_file_path: str, periodes : List[Periode]):
        
        ziparchive = zipfile.ZipFile(zip_file_path)

        features = json.load(ziparchive.open("orgUnits.geojson"))
        
        start_data = '2019-01-01'
        end_date = '2023-01-02'

        collection = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').select(band).filterDate(start_data, end_date)

        featureCollection = ee.FeatureCollection(features)

        

        
        eeScale = collection.first().select(0).projection().nominalScale()
        
        eeReducer = ee.Reducer.mean()

        #eeReducer = ee.Reducer[reducer]()

        reduced = collection.map(lambda image: 
            image.reduceRegions(
                collection=featureCollection,
                reducer=eeReducer,
                scale=eeScale
            ).map(lambda feature: 
                ee.Feature(
                    None, 
                    {
                        'ou': feature.get('id'),
                        'period': image.date().format('YYYYMMdd'),
                        'value': feature.get(reducer)
                    }
                )
            )
        ).flatten()

        valueCollection = ee.FeatureCollection(reduced)

        #print(featureCollection)
        info = valueCollection.toList(5_000).getInfo()

        # Save the info to a local file

        
 
        d = self.dataParser(info)
        print(d)

        

        #info.map(lambda e: print(e))

        #parsed = self.dataParser(info)

        #ee.List.iterate(self.dataParser)

        #print(info)
        #parsed = self.dataParser(e)
        #print(parsed)


        

        #print(valueCollection)

        

        return {"valueCollection" : ""}

        #TODO find out, what is this doing?
        #eeReducer = ee.Reducer[reducer]()





        