import datetime
import json
from typing import Iterable, List
import ee
import dataclasses
import enum
import os
from array import array
import zipfile

from pydantic import BaseModel

from climate_health.datatypes import ClimateData
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
from climate_health.time_period.date_util_wrapper import PeriodRange, TimePeriod

@dataclasses.dataclass
class Result:
    orgUntId: str
    dhis2Periode : str
    value : str

class AggregationPeriode(enum.Enum):
    MONTHLY = 'CMWF/ERA5_LAND/MONTHLY_AGGR'
    DAILY = 'ECMWF/ERA5_LAND/DAILY_AGGR'

band = "temperature_2m"
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

class GoogleEarthEngine(BaseModel):

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
    def fetch_data_climate_indicator(self, zip_file_path: str, periodes : Iterable[TimePeriod]) -> SpatioTemporalDict[ClimateData]:
        
        
        ziparchive = zipfile.ZipFile(zip_file_path)

        

        features = json.load(ziparchive.open("orgUnits.geojson"))
        
        start_date = '2019-01-01'
        end_date = '2024-02-04'

        collection = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').select(band).filterDate(start_date, end_date)

        featureCollection = ee.FeatureCollection(features)


        #the first periode
        #periode = periodes[0]


        
        #Get all days from start to end of traning periode
        #days = ee.Date(end_date).difference(ee.Date(start_date), "days")
        #Exlude the last day
        #periodList = ee.List.sequence(0, days.subtract(1))

        #print(days.getInfo())
        #print(periodList.getInfo())
        #return


        #periodeList = ee.List(list(map(lambda p: ee.Dictionary({"id" : "0", "start_date" : p.start_timestamp.date.timestamp(), "end_date" : p.end_timestamp.date.timestamp()}), periodes)))
        periodeList = ee.List([ee.Dictionary({"id" : "0", "start_date" : p.start_timestamp.date, "end_date" : p.end_timestamp.date}) for p in periodes])

        #periodeList = ee.List(simple_periode)
        
        #print(periodeList.getInfo())
       

        #periodeList = ee.List([])

        #for p in periodes:
        #    ee.List([]).add(ee.Dictionary({"id" : "0", "start_date" : "p.start_timestamp.date.timestamp()", "end_date" : "p.end_timestamp.date.timestamp()"}).getInfo())
        #print(d.getInfo())
            #ee.List.add(d)
        #print(ee.List([]).getInfo())

        #print(periodeList.getInfo())

       #def doThings(p : ee.Dictionary):
        #    p = ee.Dictionary(p)
       #     return ee.Dictionary({"to": p.get("id"), "start_date": p.get("start_date"), "end_date": p.get("end_date")})
       #     #return p2.rename("id", "to")

       # newlist =periodeList.map(doThings)

        #ee.collection
        eeScale = collection.first().select(0).projection().nominalScale()
        
        eeReducer = ee.Reducer.mean()

        def getPeriode(p):
            p = ee.Dictionary(p)
            start = ee.Date(p.get("start_date"))
            end = ee.Date(p.get("end_date"))
            filtered : ee.ImageCollection = collection.filter(ee.Filter.date(start, end))

            

            return filtered.mean().set("system:index", start_date.format("YYYYMMdd")).set("system:time_start", start.millis()).set("system:time_end", end.millis())



        


        #images_collection = periodeList.map(lambda i: getPeriode(periodes[i].start_timestamp.date, periodes[i].end_timestamp.date, collection), periodes)
        #print(images_collection.getInfo())
        #images = ee.ImageCollection(images_collection).toList(len(images_collection))
 

        #images = ee.List()
        #print(images)

        #return


        dailyCollection : ee.ImageCollection = ee.ImageCollection.fromImages(
            #DaysList containt all images from the start, to the end of the training periode
            periodeList.map(getPeriode)
            
            #periodes.map(lambda period: getPeriode(period, collection))
            #getPeriode(periode, collection)
            #periodes.map(lambda period: 
            #    getPeriode(period, collection)
            #)
        ).filter(ee.Filter.listContains("system:band_names", band))  # Remove empty images


        #print(dailyCollection.getInfo())
        #return
        #eeReducer = ee.Reducer[reducer]()


        reduced = dailyCollection.map(lambda image: 
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



        valueCollection : ee.FeatureCollection = ee.FeatureCollection(reduced)

        

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





        