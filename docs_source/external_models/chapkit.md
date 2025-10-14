# Running models with chapkit

Chapkit is a new experimental way of integrating and running models through chap.

In contrast to the current chap implementation, where models are run directly through docker and pyenv wrappers inside chap, chapkit models are separate REST API services that chap interacts with through HTTP requests.

This has several advantages:

- There are no docker-in-docker problems where chap (which is inside docker) has to run docker commands to start model containers.
- We have a strict and clear API for how chap interacts with models, which makes it easier to develop and maintain.
- Models can easily be distributed through docker images or other means, as long as they implement the chapkit API. It is up to the model to define how it is deployed and run, but chapkit makes it easy to spin up a rest api and make a docker image for a model.

This is still experimental and under development, but we have a working prototype with a few models already.

This document describes very briefly how to make a model compatible with chapkit, and how to use chapkit models in chap.


## How to make a model compatible with chapkit

This guide is not written yet, for now we refer to the chapkit documentation: [https://winterop-com.github.io/chapkit/](https://winterop-com.github.io/chapkit/)


## How to run a chapkit model from the command line with chap evaluate

To test that the model is working with chap, you can use the `chap evaluate` command. Instead of a github url or model name, you simply specify the REST API url to the model.

**Example:**

First save this model configuration to a file called `testconfig.yaml`:

```yaml
user_option_values:
  max_epochs: 2
```

Then start the chtorch command on port 5001:

```bash
docker run -p 5001:8000 ghcr.io/dhis2-chap/chtorch:chapkit2-eda20a1
```

Then we can run the following command to evaluate the model. Note the http://localhost:5001 url:

```bash
chap evaluate --model-name http://localhost:5001 --dataset-name ISIMIP_dengue_harmonized --dataset-country vietnam --report-filename report.pdf --debug --n-splits=2 --model-configuration-yaml testconfig.yaml --prediction-length 3
```


## How to use chapkit models in chap with the modeling app

Step 1: Make sure your model REST API is running on localhost. Here we assume you have a model running at http://localhost:5001. Check that the health endpoint works by going to http://localhost:5001/api/v1/health in a web browser.

In the model template configuration in your chap installation (chap-core/config/configured_models/local_config.yaml), simply add the REST api url like this:

```yaml
- url: http://localhost:5001
```

