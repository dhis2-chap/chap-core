# Local installation of a DHIS2 instance with the DHIS2 Modeling app

If you want to test chap-core with the Modeling app, follow these steps to set up a local installation of DHIS2.

### Option 1 (recommended): DHIS2 with a demo database

[docker-dhis2-core](https://github.com/dhis2-chap/docker-dhis2-core) is a ready-made Docker stack that brings up DHIS2 with a climate demo database. Clone the repository, then choose **either A or B**, depending on whether you want the stack to run chap-core for you:

- **A. Let the stack run chap-core:** run `make start-chap`. This starts DHIS2 and chap-core together, and is all you need to open the Modeling app.
- **B. Run chap-core yourself:** run `make start`. This starts DHIS2 only, and expects a chap-core you have started yourself on port 8000 (see [First-time Setup](../modeling-app/fresh-installation.md)). Choose this if you are working on the chap-core code. Until chap-core is running, the Modeling app cannot connect to CHAP.

With either option, the demo database is downloaded and loaded automatically on first start, analytics is run for you, and the DHIS2 Route that points DHIS2 at chap-core is registered automatically (so you can skip the manual URL step below).

### Option 2: a plain DHIS2 instance without test data

- [Follow these instructions](https://developers.dhis2.org/docs/cli) to install the DHIS2 cli tools
- Spin up a DHIS2 instance by running `d2 cluster up 2.41 --db-version 2.41` ([More details here](https://developers.dhis2.org/docs/cli/cluster)). Change the version number with whatever version you want.

## After DHIS2 is up

Whichever option you chose, you should now have a DHIS2 instance running at localhost:8080.

- Go to that url in your webbrowser and log in (for test: `username: admin password: district`).
- First install the `App Management` app, then install the app called `Modeling` through the App Hub.
- In the Modeling app, you will be told to put in an url to Chap. Since DHIS2 runs through a Docker container, it cannot reach Chap via `localhost`, so you need a URL that points from inside the container back to Chap running on your host machine:
  - **On Mac and Windows**, use `http://host.docker.internal:8000/**` -- Docker Desktop resolves this hostname to your host automatically.
  - **On Linux**, `host.docker.internal` is not available by default, so you need the IP of your local computer. Find it by running `ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}'` in your terminal (you may have to install ifconfig). Put `http://` before that IP and `:8000/**` after, e.g. `http://172.18.0.1:8000/**`.

## Running an evaluation

Once DHIS2 and the Modeling app are up, you can run an evaluation. This requires that:

- chap-core is up and running (see [Installation](../modeling-app/installation.md)). Check that it launched successfully by opening `http://127.0.0.1:8000/health`, which should return `{"status":"success","message":"healthy"}`.
- The Modeling app is configured with the correct URL for chap-core (the previous step).

Then, in the Modeling app:

- Go to **Evaluate**, then **Overview**, and click **New evaluation**.
- Select **weekly** as the period type.
- Set the **from** period to some week in 2022 and the **to** period to some week in 2024.
- Select a source for **precipitation** and one for **air temperature**, and select the model you want to run.
- Click **Start dry run**. This validates the configuration and data without running the full evaluation.
- If the dry run completes without errors, click **Start import** to trigger the evaluation.

On the **Overview** page you should see the evaluation running. Once it has finished, click on it to see the details.
