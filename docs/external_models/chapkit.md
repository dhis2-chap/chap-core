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

To test that the model is working with chap, you can use the `chap evaluate` command. Instead of a github url or model name, you simply specify the REST API url to the model and add --is-chapkit-model to the command to tell chap that the model is a chapkit model.

**Example:**

First save this model configuration to a file called `testconfig.yaml`:

```yaml
user_option_values:
  max_epochs: 2
```

Then start the chtorch command on port 5001:

```bash
docker run -p 5001:8000 ghcr.io/dhis2-chap/chtorch:chapkit2-d3fc2dd
```

Then we can run the following command to evaluate the model. Note the http://localhost:5001 url, which tells chap to look for the model at that url.

```bash
chap evaluate --model-name http://localhost:5001 --dataset-name ISIMIP_dengue_harmonized --dataset-country vietnam --report-filename report.pdf --debug --n-splits=2 --model-configuration-yaml testconfig.yaml --prediction-length 3 --is-chapkit-model
```


## How to use chapkit models in chap with the modeling app

NOTE: This is experimental, and the way this is done might change in the future.

1) Add your model to compose.yml, pick a port for your model that is not used by other models
2) Add your model to a config file inside config/configured_models (e.g. config/configured_models/local_config.yaml), with the url pointing to your model, using the container name you specified in compose-models, e.g. http://chtorch:5001 (localhost will not work for communication between the chap worker container and your model container). Here is an example:
  ```yaml
- url: http://chtorch:8000
  uses_chapkit: true
  versions:
    v1: "/v1"
  configurations:
    debug:
      user_option_values:
          max_epochs: 2
```
3) Start both chap and your model by running `docker compose -f compose.yml -f compose-models.yml up --build --force-recreate`

Now, the model should show up in the modeling app.
