import logging

from chap_core.dhis2_interface.src.Config import (
    DHIS2AnalyticRequest,
    ProgramConfig,
)
from chap_core.dhis2_interface.src.HttpRequest import get_request_session

logger = logging.getLogger(__name__)


def pull_analytics_elements(requestConfig: DHIS2AnalyticRequest, programConfig: ProgramConfig):
    # initilize the http client for pull job
    session = get_request_session(programConfig)

    url = f"{programConfig.dhis2Baseurl}/api/40/analytics?dimension=dx:{requestConfig.dataElementId},pe:{requestConfig.periode},ou:{requestConfig.organisationUnit}&displayProperty=NAME"

    print(url)

    print(
        f"- fetching analytics for dataElementId {requestConfig.dataElementId} for orgUnit {requestConfig.organisationUnit} for periode {requestConfig.periode}..."
    )
    try:
        response = session.get(url)
    except Exception as e:
        logger.error("Could not get data from dhis 2: %s", e)
        raise
    if response.status_code != 200:
        raise Exception(f"Could not fetch data. \nError code: {response.status_code}")
    print(
        f"- 200 OK - fetched analytics for dataElementId {requestConfig.dataElementId} for periode {requestConfig.periode}"
    )
    response_json = response.json()
    return response_json
