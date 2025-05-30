# Configure DHIS2 Modeling App and Chap Core to work together

**Requirements:**

- Minimum DHIS2 version: **2.40.7**
- Access to a server running DHIS2

The Modeling App is dependent on connecting to a Chap server, as running the models is a process that needs to be handled by Chap Core. In this tutorial, we will deploy Chap Core to the same server that DHIS2 is running on, and making the Chap Core REST API available internally to the DHIS2 backend. The Modeling App will then use the DHIS2 Route API to connect to the Chap Core endpoint. We will explain the purpose of the Route API later.

**IMPORTANT:** We do not want to make the Chap Core endpoint publicly available on the internet, as Chap Core does not have any method of authenticating requests.

**NOTE:** Previously, Chap Core required you to configure it with Google Earth Engine Credentials. This is not needed anymore, since the Modeling App is now using climate data imported into DHIS2 by the [DHIS2 Climate App](https://apps.dhis2.org/app/effb986c-a3c7-485e-a2f6-5e54ff9df7c3). Using the DHIS2 Climate App requires you to set up [DHIS2 with Google Earth Engine](https://docs.dhis2.org/en/topics/tutorials/google-earth-engine-sign-up.html)

## Recommendation around containerization

We strongly recommend using Chap Core with a container framework such as LXC, where Chap Core runs in its own environment. Chap Core consists of several different services. In our [Chap Core repo](https://github.com/dhis2-chap/chap-core), we provide a [docker-compose file](https://github.com/dhis2-chap/chap-core/blob/master/compose.yml) that containerizes each of these services and makes them work together. We highly recommend you use this docker-compose file when installing Chap Core, since installing each of these services could be very difficult without Docker.

## LXC container setup

To run Chap Core on a server that is using LXC, you need to create a new LXC container dedicated to Chap Core. Within this LXC container, you then need to install Docker. Ubuntu has documentation on how to install Docker within an LXC container located at [https://ubuntu.com/tutorials/how-to-run-docker-inside-lxd-containers#1-overview](https://ubuntu.com/tutorials/how-to-run-docker-inside-lxd-containers#1-overview)

The Chap Core team has an example of a Chap Core LXC setup, which can be found at [https://github.com/dhis2-chap/infrastructure](https://github.com/dhis2-chap/infrastructure). Handle this code with care. The code is used to deploy Chap Core on a server for conducting integration testing. We recommend everyone only use this as an example and create their own bash script for deploying Chap Core within an LXC container.

## Clone the Chap Core repo into your LXC container

Within your LXC container, you need to clone the Chap Core repo. Information about **how to get** and **start Chap Core**, is located at [https://github.com/dhis2-chap/chap-core/releases/](https://github.com/dhis2-chap/chap-core/releases/)

After you have started Chap Core, the Chap Core REST API will be available at port 8000.

## Identify the Chap Core server private IP address and verify Chap Core is running properly

We now have Chap Core running in Docker within an LXC container dedicated to Chap Core. Next, you need to identify the private IP address (for instance 192.168.0.174) of the LXC container running Chap Core. This IP address is needed to get the DHIS2 Route API to work. **Again, the IP address of Chap Core should not be exposed publicly, only internally on the server.** If you are running with LXC, you could, for instance, use **lxc list** to locate this IP address. You should then use the "curl" command to verify that you can connect to the LXC container. In the same terminal as you listed your containers, try to run `curl http://[YOUR_IP_ADDRESS]:8000/docs` (for instance `curl http://192.168.0.174:8000/docs`). If Chap Core is running correctly, you should, in response, get some HTML swagger content.

Next, verify if you can connect to Chap Core from the container you are running DHIS2 by executing this container and using the curl command (for instance `curl http://192.168.0.174:8000/docs`).

## Install the DHIS2 Modeling App

Go into **App Management** in your DHIS2 instance, and install the [Modeling App](https://apps.dhis2.org/app/a29851f9-82a7-4ecd-8b2c-58e0f220bc75)

## Route API

The newer version of DHIS2 supports a built-in reverse proxy named [Routes API](https://docs.dhis2.org/en/develop/using-the-api/dhis-core-version-242/route.html). This means you could use the DHIS2 backend to forward a request (typically sent from a frontend application) to another IP address (in our case the Chap Core). To use the Route API (the reverse proxy), we need to configure a "route" in the DHIS2 backend that forwards requests sent from the Modeling App to the Chap Core. First, we create a Route in DHIS2, which means a resource in DHIS2 that holds information about which IP our "route" in DHIS2 should forward requests to.

The Modeling App supports a user interface for creating the needed route, where you only need to specify which IP the request should be forwarded to by the route. To create a new route, it requires you to have the DHIS2 System Administrator Role. In the form where you create the route, you need to speficy the IP adress (for instance `http://192.168.0.174/**`) you located (and verified that was accessible from the DHIS2 container) in the step above.

**IMPORTANT:** You need to configure the route as a "wildcard route", by ending the IP with `/**` More information about Wildcard routes could be found [here](https://docs.dhis2.org/en/develop/using-the-api/dhis-core-version-242/route.html#wildcard-routes)

## Verifying the route is working in the Modeling App

You could now go to the Settings page in the Modeling App and verify if the Modeling App can connect to Chap Core. If not, check if the IP address configured in the route is correct and corresponds with the IP you used when you tried to connect to Chap Core from the container you are running DHIS2 in.
