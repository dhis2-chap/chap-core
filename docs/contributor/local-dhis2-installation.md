# Local installation of a DHIS2 instance with the DHIS2 Modeling app

If you want to test chap-core with the Modeling app, follow these steps to set up a local installation of DHIS2.

We have an internal database that can be used to set up a DHIS2 instance with testdata. If you are an internal developer, you will have access to this through our internal drive. Follow these steps (if you don't have access to this database, and want to set up a general instance, see steps below):

- Download the zip-file, unzip it and run `docker compose up` in the unzipped directory.
  - Note: If you are on linux, you will have to edit the docker-compose.yaml file and change `platform` to `linux/amd64`.
  - Note: You may have to restart the web docker container if this started before the db container was up.
- Run analytics by opening Data administration, go to analytics tables, uncheck all boxes and click "Start export"

To set up a DHIS2 instance without this test db, do the following:

- [Follow these instructions](https://developers.dhis2.org/docs/cli) to install the DHIS2 cli tools
- Spin up a DHIS2 instance by running `d2 cluster up 2.41 --db-version 2.41` ([More details here](https://developers.dhis2.org/docs/cli/cluster)). Change the version number with whatever version you want.

After following any of the guides above, you should have a DHIS2 instance running at localhost:8080.

- Go to that url in your webbrowser and log in.
- First install the `App Management` app, then install the app called `Modeling` through the App Hub.
- In the Modeling app, you will be told to put in an url to Chap. Since DHIS2 runs through a Docker container, you will need to put in an IP to your local computer. This ip can be found by running `ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}'` in your terminal (you may have to install ifconfig). Put `http://` before that IP and `:8000/**` after, e.g. `http://172.18.0.1:8000/**`.
