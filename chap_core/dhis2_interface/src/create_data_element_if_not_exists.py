from chap_core.dhis2_interface.src.Config import ProgramConfig
from chap_core.dhis2_interface.src.HttpRequest import get_request_session
import logging

logger = logging.getLogger(__name__)


def create_data_element_if_not_exists(config: ProgramConfig, code_prefix: str, dict: dict[str, str], disease=""):
    for key, value in dict.items():
        code = f"{code_prefix}_{key}".upper()
        # first check if exisits, set hash if exists
        dict[key] = __get_elements_if_exisits(config, code)

        # if not found, create new dataElement in DHIS2
        if dict[key] is None:
            # create the data element
            dict[key] = __create_data_element(programConfig=config, code=code, quantile=key, disease=disease)

    # return the updated dict, with code and hashes
    print(dict)


def __get_elements_if_exisits(programConfig: ProgramConfig, code: str):
    session = get_request_session(programConfig)
    url = f"{programConfig.dhis2Baseurl}/api/dataElements.json?filter=code:eq:{code}"

    try:
        response = session.get(url)
    except Exception as e:
        logger.error("Could not get data from dhis 2: %s", e)
        raise
    if response.status_code != 200:
        raise Exception(f"Could not fetch data. \nError code: {response.status_code}")
    # return None if dataElement does not exisits
    if len(response.json()["dataElements"]) == 0:
        return None
    print(f"- 200 OK - dataElement with code '{code}' already exists, skipping creation.")
    uid = response.json()["dataElements"][0]["id"]
    return uid


def __create_data_element(programConfig: ProgramConfig, code: str, quantile: str, disease: str):
    session = get_request_session(programConfig)

    name = f'CHAP - {disease.lower()} {quantile.lower().replace("_", " ")}'

    body = {
        "aggregationType": "COUNT",
        "code": code,
        "domainType": "AGGREGATE",
        "description": "DataElement created by CHAP for {disease} with quantile {quantile}",
        "valueType": "NUMBER",
        "name": name,
        "shortName": name,
        "aggregationLevels": [3, 2, 4, 1],
    }

    url = f"{programConfig.dhis2Baseurl}/api/dataElements"

    try:
        response = session.post(url=url, json=body)
    except Exception as e:
        logger.error(f"Could not create {name} to dhis 2: %s", e)
        raise
    if response.status_code != 201:
        raise Exception(
            f'Could not create. \nError code: {response.status_code}\nmessage: {response.json()["message"]}'
        )
    print(f"- 201 OK - dataElement with code '{code}' created")
    uid = response.json()["response"]["uid"]
    return uid
