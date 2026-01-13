# Setting up Chap REST-API locally

This is a short example for how to setup Chap-core locally as a service using docker-compose.

**Requirements:**

- Docker is installed **and running** on your computer (Installation instructions can be found at [https://docs.docker.com/get-started/get-docker/](https://docs.docker.com/get-started/get-docker/)).

## Step-by-Step Instructions:

1. Clone the Chap core repo by running `git clone https://github.com/dhis2-chap/chap-core.git`

2. Run the docker compose file with `docker compose -f compose.yml up`. The first time you do this, it can take a few minutes to finish. Once it's completed, it should have created the following docker services:

   - `redis` for receiving and queueing job requests
   - `worker` for executing the incoming work requests from queue
   - `chap` containing the main functionality and the rest-api
   - `postgres` for storing chap-related data

3. Check that the chap rest api works by going to http://localhost:8000/docs
