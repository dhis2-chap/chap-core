import sys

from chap_core.datatypes import HealthData
from chap_core.dhis2_interface.json_parsing import (
    parse_climate_data,
    parse_disease_data,
    parse_population_data,
    predictions_to_datavalue,
)
from chap_core.dhis2_interface.src.PushResult import push_result
from chap_core.dhis2_interface.src.create_data_element_if_not_exists import (
    create_data_element_if_not_exists,
)
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.dhis2_interface.src.PullAnalytics import pull_analytics_elements
from chap_core.dhis2_interface.src.Config import (
    DHIS2AnalyticRequest,
    ProgramConfig,
)


class ChapPullPost:
    def __init__(self, dhis2Baseurl, dhis2Username, dhis2Password):
        self.config = ProgramConfig(
            dhis2Baseurl=dhis2Baseurl,
            dhis2Username=dhis2Username,
            dhis2Password=dhis2Password,
        )

        self.getDHIS2PullConfig()

    def getDHIS2PullConfig(self):
        # Some data here that should be retrived from DHIS2, for example trough the dataStore-API. We need dataElementId, periode and organisationUnit, for now - just hardcoded.

        # siera leone all level LEVEL-wjP19dkFeIk
        # laos all level LEVEL-qpXLDdXT3po

        # dataElementId here is "IDS - Dengue Fever (Suspected cases)"
        # orgUnit would fetch data for each 17 Laos provinces
        # periode is what it is
        self.DHIS2HealthPullConfig = DHIS2AnalyticRequest(
            dataElementId="cYeuwXTCPkU",
            organisationUnit="LEVEL-wjP19dkFeIk",
            periode="LAST_12_MONTHS",
        )
        self.DHIS2PopulationPullConfig = DHIS2AnalyticRequest(
            dataElementId="WUg3MYWQ7pt",
            organisationUnit="LEVEL-wjP19dkFeIk",
            periode="TODAY",
        )
        self.DHIS2ClimatePullConfig = DHIS2AnalyticRequest(
            dataElementId="hash,hash",
            organisationUnit="LEVEL-wjP19dkFeIk",
            periode="LAST_52WEEKS",
        )

    def pullPopulationData(self):
        json = pull_analytics_elements(self.DHIS2PopulationPullConfig, self.config)
        return parse_population_data(json)

    def pullDHIS2Analytics(self):
        json = pull_analytics_elements(self.DHIS2HealthPullConfig, self.config)
        return parse_disease_data(json)

    def pullDHIS2ClimateData(self):
        json = pull_analytics_elements(self.DHIS2ClimatePullConfig, self.config)
        return parse_climate_data(json)

    def startModelling(self):
        # do the fancy modelling here?
        return

    def pushDataToDHIS2(self, data: DataSet[HealthData], model_name: str, do_dict=True):
        # TODO do we need to delete previous modells?, or would we overwrite exisitng values?

        # used to prefix CHAP-dataElements in DHIS2
        code_prefix = "CHAP_" + model_name

        # dict with quantile and the hash value
        name_dict = {
            "quantile_low": "quantile_low",
            "median": "median",
            "quantile_high": "quantile_high",
        }

        # Insert hashes from DHIS2 into the dict
        if do_dict:
            create_data_element_if_not_exists(
                config=self.config,
                code_prefix=code_prefix,
                dict=name_dict,
                disease=model_name,
            )

        # create dict
        values = predictions_to_datavalue(data, name_dict)
        response, body = push_result(self.config, values)

        return body


if __name__ == "__main__":
    # validate arguments
    if len(sys.argv) != 4:
        print("UNVALID ARGUMENTS: Usage: ChapProgram.py <dhis2Baseurl> <dhis2Username> <dhis2Password>")
        sys.exit(1)

    process = ChapPullPost(
        dhis2Baseurl=sys.argv[1].rstrip("/"),
        dhis2Username=sys.argv[2],
        dhis2Password=sys.argv[3],
    )

    # set config used in the fetch request
    process.pullDHIS2Analytics()
    process.pullPopulationData()

    process.pullDHIS2ClimateData()
    process.startModelling()

    d = {"": ""}
    sp = DataSet(d)

    process.pushDataToDHIS2(sp, "dengue")
