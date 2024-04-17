import requests
import json
import logging

from climate_health.dhis2_interface.src.Config import DHIS2AnalyticRequest
from climate_health.dhis2_interface.src.HttpRequest import get_request_session

logger = logging.getLogger(__name__)

def pull_pupulation_data(requestConfig : DHIS2AnalyticRequest, programConfig):
    # initilize the http client for pull job
    session = get_request_session(programConfig)
    
    url = f'{programConfig.dhis2Baseurl}/api/40/analytics?dimension=dx:{requestConfig.dataElementId},pe:{requestConfig.periode},ou:{requestConfig.organisationUnit}&displayProperty=NAME'
    print(
        f"- fetching analytics for dataElementId {requestConfig.dataElementId} for orgUnit {requestConfig.organisationUnit} for periode {requestConfig.periode}...")
    try:
        response = session.get(url)
    except Exception as e:
        logger.error('Could not get data from dhis 2: %s', e)
        raise
    if (response.status_code != 200):
        raise Exception(f"Could not fetch data. \nError code: {response.status_code}")
    print(
        f"- 200 OK - fetched analytics for dataElementId {requestConfig.dataElementId} for periode {requestConfig.periode}")
    response_json = response.json()
    return response_json



def pull_analytics(requestConfig, programConfig):
    # initilize the http client for pull job
    session = get_request_session(programConfig)

    url = f'{programConfig.dhis2Baseurl}/api/40/analytics?dimension=dx:{requestConfig.dataElementId},pe:{requestConfig.periode},ou:{requestConfig.organisationUnit}&displayProperty=NAME'
    print(
        f"- fetching analytics for dataElementId {requestConfig.dataElementId} for orgUnit {requestConfig.organisationUnit} for periode {requestConfig.periode}...")
    try:
        response = session.get(url)
    except Exception as e:
        logger.error('Could not get data from dhis 2: %s', e)
        raise
    if (response.status_code != 200):
        raise Exception(f"Could not fetch data. \nError code: {response.status_code}")
    print(
        f"- 200 OK - fetched analytics for dataElementId {requestConfig.dataElementId} for periode {requestConfig.periode}")
    response_json = response.json()
    return response_json
