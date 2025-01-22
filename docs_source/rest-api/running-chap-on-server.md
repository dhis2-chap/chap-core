# Running CHAP Core on a server with DHIS2 to be used with the Prediction App

Before you set up CHAP Core on a server, we recommend you familiarize yourself with CHAP Core, the Prediction App, and test it out locally.

**Requirements:**
  - Access to credentials to Google Earth Engine (Google Service Account Email and Private Key) Read more here: [https://docs.dhis2.org/en/manage/reference/google-service-account-configuration.html](https://docs.dhis2.org/en/manage/reference/google-service-account-configuration.html)
  - DHIS2 version **2.41.2+**

In the documentation for [Use Prediction App to transfer data directly to CHAP Core’s REST-API](https://dhis2-chap.github.io/chap-core/prediction-app/prediction-app.html), we were running CHAP Core locally and exposed the CHAP Core on http://localhost:8000. In that example, we configured the Prediction App to connect to CHAP Core over http://localhost:8000 by editing the CHAP-Core backend server URL. Data were then transferred directly from the Prediction App to CHAP Core over http://localhost:8000. This is obviously not a recommended solution in a production environment, since it will require the user to have CHAP Core running locally on their machine. Therefore, it makes sense to deploy CHAP to a server. To achieve this, we will deploy CHAP Core to the same server as a DHIS2 backend is running, ensuring that the CHAP Core REST-API is available internally to the DHIS2 backend. The Prediction App will then use the DHIS2 Route API to connect to the CHAP Core endpoint. We will explain the purpose of the Route API later.

**IMPORTANT:** We do not want to make the CHAP Core endpoint publicly available on the internet, since CHAP Core does not have any way of authenticating requests.

## Recommendation around containerization

We strongly recommend using CHAP Core with a container framework such as Docker, LXC, or Kubernetes, where CHAP Core runs in its own container. CHAP Core consists of several different services. In our [CHAP Core repo](https://github.com/dhis2-chap/chap-core), we provide a docker-compose file that containerizes each of these services and makes them work together. We highly recommend you use this docker-compose file when installing CHAP Core since installing each of these services could be difficult without Docker.

## LXC container setup
To run CHAP Core on a server that is using LXC, you need to create a new LXC container dedicated to CHAP Core. Within this LXC container, you then need to install Docker. The CHAP Core team has an example of such a setup, which can be found at [https://github.com/dhis2-chap/infrastructure](https://github.com/dhis2-chap/infrastructure). Handle this code with care. The code is used to deploy CHAP Core on a server for conducting integration testing. We recommend everyone only uses this as an example and creates their own bash script for deploying CHAP Core within an LXC container.

## Clone the CHAP Core repo into your LXC container
Within your LXC container dedicated to CHAP Core, you need to use git to clone the CHAP Core repo by using **git clone https://github.com/dhis2-chap/chap-core.git**. If you follow the [infrastructure repo](https://github.com/dhis2-chap/infrastructure), you can see that we fetch the CHAP Core repo and start CHAP Core immediately after the LXC container is created. If you do so, you will need to store your Google Earth Engine credentials outside the LXC container as environment variables. See these two lines, for how to insert environment variables into the LXC container.

```bash
  lxc config set chap-core environment.GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY "$GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY"
  lxc config set chap-core environment.GOOGLE_SERVICE_ACCOUNT_EMAIL "$GOOGLE_SERVICE_ACCOUNT_EMAIL"
```

## Add Google Earth Engine credentials directly to the container
If you have not stored the Google Earth Engine credentials outside the LXC container, you have the option to add them directly within the LXC container. This is not recommended since if your LXC container gets destroyed, you will need to manually add the credentials back again. Instead, we recommend you to have an "infrastructure as code" approach, where you, for instance, have a pipeline for deploying CHAP Core. If you do not have the pipeline yet and only want to test CHAP Core out, you have the option to add credentials directly in the LXC container. To achieve this, you could create a **.env** file in the root folder in the CHAP Core repo, where you insert the variables **"GOOGLE_SERVICE_ACCOUNT_EMAIL"** and **"GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY"**.

The **.env** file should look similar to the following content:

```bash
  GOOGLE_SERVICE_ACCOUNT_EMAIL="your-google-service-account@company.iam.gserviceaccount.com"
  GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----<your-private-key>-----END PRIVATE KEY-----"
```

The file should be stored at the same level where you find the Docker compose file, such as below:

```
/chap-core
|-- compose.integration.test.yml
|-- compose.test.yml
|-- compose.yml
|-- .env
|-- chap_core/
```
*Not all files are included in this file-structure snippet*

**NB!** If you were already running CHAP Core when discovering that these variables were missing, you need to run **docker compose down** before starting CHAP Core with **docker compose up**. Warnings about missing environment variables should disappear.

## Start CHAP Core
After you have cloned the CHAP Core repo, you start CHAP Core by running **docker compose up**. CHAP Core will then expose the REST Api at port 8000.

## Identify the CHAP Core server IP address and verify CHAP Core is running properly
We now have CHAP Core running in Docker within an LXC container dedicated to CHAP Core. Next, you need to identify the IP address (for instance 192.168.0.174) of the LXC container running CHAP Core. This IP address is needed to get the DHIS2 Route API to work. Again, the IP address of CHAP Core should not be exposed publicly, only internally on the server. If you are running with LXC, you could, for instance, use **lxc list** to locate this IP address. You should then use the "curl" command to verify that you can connect to the LXC container. In the same terminal as you listed your containers, try to run **curl http://[YOUR_IP_ADDRESS]:8000/docs** (for instance curl http://192.168.0.174:8000/docs). If CHAP Core is running correctly, you should, in response, get some HTML swagger content. Next, verify if you can connect to CHAP Core from the container you are running DHIS2 by executing this container and using the curl command (for instance curl http://192.168.0.174:8000/docs).

## Route API
The newer version of DHIS2 supports a built-in reverse proxy named [Routes API](https://docs.dhis2.org/en/develop/using-the-api/dhis-core-version-241/route.html). This means you could use the DHIS2 backend to forward a request (typically sent from a frontend application) to another IP address or URL (in our case the CHAP Core). To use the Route API (the reverse proxy), we need to configure a "route" in the DHIS2 backend that forwards requests sent from the Prediction App to the CHAP Core. First, we create a Route in DHIS2, which means a resource in DHIS2 that holds information about which URL/IP our "route" in DHIS2 should forward requests to. The Prediction App supports a user interface for creating the needed route, where you only need to specify which URL or IP the request should be forwarded to by the route. To create a new route, it requires you to have the DHIS2 System Administrator Role. The IP/URL you need to insert is the IP/URL (for instance http://192.168.0.174) you located (and verified that was accessible from the DHIS2 container) in the step above. Insert this IP/URL into the input field and click "Save". Additionally, clear any value in the "Edit CHAP-Core URL" dialog (it should be empty when using the Route API), and click save.

## Using The Prediction App
The Prediction App should now be able to use the "route" you created to connect to CHAP Core. You could now go to the "Route API settings" in the Prediction App and verify if the Prediction App can connect to CHAP Core. If not, check if the "route" values are correct and correspond with the IP/URL you used when you tried to connect to CHAP Core from the container you are running DHIS2 in. 

Prediction App Video: [https://youtu.be/6VYws5ywxtg](https://youtu.be/6VYws5ywxtg)

