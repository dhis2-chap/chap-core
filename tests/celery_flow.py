"""
This is meant to be a standalone python file for testing the flow with docker compose

This file should be possible to run without chap installed, only with the necessary requirements
"""
import json
import time

import requests
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

hostname = 'chap'
chap_url = "http://%s:8000" % hostname

def main():
    add_url = chap_url + "/v1/debug/add-numbers?a=1&b=2"
    print(add_url)
    ensure_up(chap_url)
    try:
        response = requests.get(add_url)
    except:
        print("Failed to connect to %s" % chap_url)
        logger.error("Failed when fetching models")
        print("----------------Exception info----------------")
        exception_info = requests.get(chap_url + "/v1/get-exception").json()
        print(exception_info)
        logger.error(exception_info)
        logger.error("Failed to connect to %s" % chap_url)
        raise

    for i in range(5):
        time.sleep(2)
        status = requests.get(chap_url + "/v1/debug/get-status") #?task_id=" + response.json()["task_id"])
        if status.status_code not in (200, 404):
            print(status)
            break
        if status.json()["status"] == "SUCCESS":
            print(status.json())


def ensure_up(chap_url):
    for _ in range(5):
        try:
            requests.get(chap_url + "/v1/status")
            break
        except requests.exceptions.ConnectionError:
            time.sleep(5)


if __name__ == "__main__":
    main()
    #

#evaluate_model(chap_url, dataset(), {"name": "naive_model"})
