# Setting up CHAP REST-API locally

This is a short example for how to setup CHAP-core locally as a service using docker-compose. 

**Requirements:**
- Docker is installed **and running** on your computer (Installation instructions can be found at [https://docs.docker.com/get-started/get-docker/](https://docs.docker.com/get-started/get-docker/)).

## Step-by-Step Instructions:

1. Locate the folder containing your local copy of the CHAP Core codebase. This depends if you installed CHAP Core as a [commandline tool](chap-core-setup), as a [git repository if you're a contributor](chap-contributor-setup), as a [local Docker container](docker-compose-doc), or [on a server](running-chap-on-server). 

2. Make sure that you have added the Google Earth Engine credentials to your CHAP Core folder, as [described here](earth-engine-auth). 

3. On the commandline, go to the chap-core directory with `cd path/to/chap-core`. 

4. Then run the docker compose file with `docker compose -f compose.yml --env-file=.env up`. The first time you do this, it can take a few minutes to finish. Once it's completed, it should have created the following docker services:
    - `redis` for receiving and queueing job requests
    - `worker` for executing the incoming work requests from queue
    - `chap` containing the main functionality and the rest-api
    - `postgres` for storing chap-related data

5. Check that the chap rest api works by going to http://localhost:8000/docs
