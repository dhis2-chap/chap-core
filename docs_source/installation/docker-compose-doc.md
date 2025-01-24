# Setting up CHAP REST-API locally

This is a short example for how to setup CHAP-core locally as a service using docker-compose. 

1. Go to the chap-core directory with `cd path/to/chap-core`

2. Create a `.env` file with the following env variables to authenticate with google earth engine (refer to ... for how to get this information): 
    
        GOOGLE_SERVICE_ACCOUNT_EMAIL=...
        GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY=...

3. Then run the docker compose file with `docker compose -f compose.yml --env-file=.env up`. The first time you do this, it can take a few minutes to finish. Once it's completed, it should have created the following docker services:
    - `redis` for receiving and queueing job requests
    - `worker` for executing the incoming work requests from queue
    - `chap` containing the main functionality and the rest-api
    - `postgres` for storing chap-related data

4. Check that the chap rest api works by going to http://localhost:8000/docs
