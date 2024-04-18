
import dataclasses
from typing import List
from climate_health.dhis2_interface.src.Config import ProgramConfig
from climate_health.dhis2_interface.src.HttpRequest import get_request_session
import logging

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class DataValue:
    value:int
    orgUnit:str
    dataElement:str
    period:str
    
def push_result(programConfig : ProgramConfig, dataValues : List[DataValue]):
    session = get_request_session(programConfig)

    body = {
        "dataValues" : dataValues
    }
    
    url = f'{programConfig.dhis2Baseurl}/api/40/dataValueSets'
    
    print(
        f"- posting CHAP result...")
    try:
        response = session.post(url=url, data=body)
    except Exception as e:
        logger.error('Could not post CHAP-result to dhis 2: %s', e)
        raise
    if (response.status_code != 201):
        raise Exception(f"Could not fetch data. \nError code: {response.status_code}")
    print(
        f"- 200 OK - fetched analytics for dataElementId {requestConfig.dataElementId} for periode {requestConfig.periode}")
    response_json = response.json()
    return response_json