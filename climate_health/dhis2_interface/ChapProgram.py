import sys

from climate_health.dhis2_interface.json_parsing import parse_disease_data, parse_population_data
from src.PullAnalytics import pull_analytics
from src.PullAnalytics import pull_pupulation_data
from src.Config import DHIS2AnalyticRequest, ProgramConfig


class ChapPullPost:
    def __init__(self, dhis2Baseurl, dhis2Username, dhis2Password):
        self.config = ProgramConfig(dhis2Baseurl=dhis2Baseurl, dhis2Username=dhis2Username, dhis2Password=dhis2Password)
        
        self.getDHIS2PullConfig()


    def getDHIS2PullConfig(self):
        # Some data here that should be retrived from DHIS2, for example trough the dataStore-API. We need dataElementId, periode and organisationUnit, for now - just hardcoded.

        # dataElementId here is "IDS - Dengue Fever (Suspected cases)"
        # orgUnit would fetch data for each 17 Laos provinces
        # periode is what it is
        self.DHIS2HealthPullConfig = DHIS2AnalyticRequest(dataElementId="wryMP8p8k1C", organisationUnit="LEVEL-qpXLDdXT3po",
                                           periode="LAST_52_WEEKS")
        self.DHIS2PopulationPullConfig = DHIS2AnalyticRequest(dataElementId="DkmMEcubiPv", organisationUnit="LEVEL-qpXLDdXT3po",
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

    def pushDataToDHIS2(self):
        # push to DHIS2
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
    process.pushDataToDHIS2()
