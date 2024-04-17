
import requests
import json

def pullAnalytics(programConfig, dhis2Config):
    #initilize the http client for pull job
    session = requests.Session()
    session.headers.update({'Accepts': 'application/json'})
    session.auth = (programConfig.dhis2Username, programConfig.dhis2Password)
    
    url = f'{programConfig.dhis2Baseurl}/api/40/analytics?dimension=dx%{dhis2Config.dataElementId},pe:{dhis2Config.periode},ou:{dhis2Config.organisationUnit}&displayProperty=NAME'
    
    print(f"- fetching analytics for dataElementId {dhis2Config.dataElementId} for orgUnit {dhis2Config.organisationUnit} for periode {dhis2Config.periode}...")
    response = session.get(url)

    if(response.status_code == 200):
        print(f"- 200 OK - fetched analytics for dataElementId {dhis2Config.dataElementId} for periode {dhis2Config.periode}")

        #Save respons to a file
        fileName = f"dhis2analyticsResponses/{dhis2Config.dataElementId}_{dhis2Config.organisationUnit}_{dhis2Config.periode}.json"
        with open(fileName, "w") as f:
            json.dump(response.json(), f, sort_keys = True, indent=4, ensure_ascii=False)
            print(f"- new file created: {fileName}")
    else:
        print(f"Could not fetch data. \nError code: {response.status_code}")
    
    return True

