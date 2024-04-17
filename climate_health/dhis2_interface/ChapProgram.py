import sys

from climate_health.dhis2_interface.json_parsing import parse_disease_data
from .src.ValidateArgs import validate
from .src.PullAnalytics import pullAnalytics, pull_analytics
from .src.Config import DHIS2PullConfig, ProgramConfig


class ChapPullPost:
    def __init__(self, dhis2Baseurl, dhis2Username, dhis2Password):
        self.config = ProgramConfig(dhis2Baseurl=dhis2Baseurl, dhis2Username=dhis2Username, dhis2Password=dhis2Password)
        self.DHIS2PullConfig = DHIS2PullConfig(dataElementId="3AwryMP8p8k1C", organisationUnit="LEVEL-qpXLDdXT3po",
                                           periode="LAST_52_WEEKS")

    def getDHIS2PullConfig(self):
        # Some data here that should be retrived from DHIS2, for example trough the dataStore-API. We need dataElementId, periode and organisationUnit, for now - just hardcoded.

        # dataElementId here is "IDS - Dengue Fever (Suspected cases)"
        # orgUnit would fetch data for each 17 Laos provinces
        # periode is what it is
        self.DHIS2PullConfig = DHIS2PullConfig(dataElementId="3AwryMP8p8k1C", organisationUnit="LEVEL-qpXLDdXT3po",
                                               periode="LAST_52_WEEKS")


    def pullDHIS2Analytics(self):
        json = pull_analytics(self.DHIS2PullConfig, self.config)
        return parse_disease_data(json)
        # pullAnalytics(self.config, self.DHIS2PullConfig)

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
    if (validate() is False):
        sys.exit(1)

    process = ChapPullPost(dhis2Baseurl=sys.argv[1].rstrip('/'), dhis2Username=sys.argv[2], dhis2Password=sys.argv[3])

    # set config used in the fetch request
    process.getDHIS2PullConfig()
    process.pullDHIS2Analytics()
    process.pullDHIS2ClimateData()
    process.startModelling()
    process.pushDataToDHIS2()
