import requests

from chap_core.dhis2_interface.src.Config import ProgramConfig


def get_request_session(programConfig: ProgramConfig):
    session = requests.Session()
    session.headers.update({"Accepts": "application/json"})
    session.headers.update({"Content-type": "application/json"})
    session.auth = (programConfig.dhis2Username, programConfig.dhis2Password)
    return session
