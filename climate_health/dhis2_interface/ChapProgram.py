import sys


from climate_health.datatypes import HealthData
from climate_health.dhis2_interface.json_parsing import parse_disease_data, parse_population_data, predictions_to_json
from climate_health.dhis2_interface.src.PushResult import DataValue, push_result
from climate_health.dhis2_interface.src.create_data_element_if_not_exists import create_data_element_if_not_exists
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
from climate_health.dhis2_interface.src.PullAnalytics import pull_analytics
from climate_health.dhis2_interface.src.PullAnalytics import pull_pupulation_data
from climate_health.dhis2_interface.src.Config import DHIS2AnalyticRequest, ProgramConfig


class ChapPullPost:
    def __init__(self, dhis2Baseurl, dhis2Username, dhis2Password):
        self.config = ProgramConfig(dhis2Baseurl=dhis2Baseurl, dhis2Username=dhis2Username, dhis2Password=dhis2Password)
        
        self.getDHIS2PullConfig()


    def getDHIS2PullConfig(self):
        # Some data here that should be retrived from DHIS2, for example trough the dataStore-API. We need dataElementId, periode and organisationUnit, for now - just hardcoded.

        #siera leone all level LEVEL-wjP19dkFeIk
        # laos all level LEVEL-qpXLDdXT3po

        # dataElementId here is "IDS - Dengue Fever (Suspected cases)"
        # orgUnit would fetch data for each 17 Laos provinces
        # periode is what it is
        self.DHIS2HealthPullConfig = DHIS2AnalyticRequest(dataElementId="cYeuwXTCPkU", organisationUnit="LEVEL-wjP19dkFeIk",
                                           periode="LAST_12_MONTHS")
        self.DHIS2PopulationPullConfig = DHIS2AnalyticRequest(dataElementId="WUg3MYWQ7pt", organisationUnit="LEVEL-wjP19dkFeIk",
                                           periode="TODAY")

    def pullPopulationData(self):
        json = pull_pupulation_data(self.DHIS2PopulationPullConfig, self.config)
        return parse_population_data(json)

    def pullDHIS2Analytics(self):
        json = pull_analytics(self.DHIS2HealthPullConfig, self.config)
        return parse_disease_data(json)

    def pullDHIS2ClimateData(self):
        # pull Climate-data from climate-data app
        return

    def startModelling(self):
        # do the fancy modelling here?
        return

    def pushDataToDHIS2(self, data : SpatioTemporalDict[HealthData], model_name : str):
        # TODO do we need to delete previous modells?, or would we overwrite exisitng values?
        
        #used to prefix CHAP-dataElements in DHIS2
        code_prefix = "CHAP_" + model_name

        #dict with quantile and the hash value
        dict = {"quantile_low": None, "median": None, "quantile_high": None}

        # Insert hashes from DHIS2 into the dict
        create_data_element_if_not_exists(config=self.config, code_prefix=code_prefix, dict=dict, disease=model_name)

        # create dict 
        values = predictions_to_json(data, dict)
        push_result(self.config, values)

        return


if __name__ == "__main__":
    # validate arguments
    if len(sys.argv) != 4:
        print("UNVALID ARGUMENTS: Usage: ChapProgram.py <dhis2Baseurl> <dhis2Username> <dhis2Password>")
        sys.exit(1)

    process = ChapPullPost(dhis2Baseurl=sys.argv[1].rstrip('/'), dhis2Username=sys.argv[2], dhis2Password=sys.argv[3])

    # set config used in the fetch request
    process.pullDHIS2Analytics()
    process.pullPopulationData()
    
    process.pullDHIS2ClimateData()
    process.startModelling()

    d = {"" : ""}
    sp = SpatioTemporalDict(d)

    process.pushDataToDHIS2(sp, "dengue")
