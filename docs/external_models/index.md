---
description: "How to make a forecasting model in R or Python work with Chap: train and predict scripts, an MLproject file, chap eval, and the Chap Modeling App."
---

<script>
if (window.location.hostname === "dhis2-chap.github.io") {
    window.location.href = "https://chap.dhis2.org/chap-modeling-platform/";
}
</script>

# Getting started as a model developer

The Chap platform brings several advantages to you as a model developer:

* It provides a [broad range of supporting functionality for modelling](supporting_functionality.md), allowing you to focus on what is unique in your project, while relying on Chap for data parsing, model tuning, rigorous model assessment and more.
* It allows you to evaluate your model [side-by-side with a range of other models](side_by_side_comparison.md) in the field
* It comes with [native connection to DHIS2 and the DHIS2 Modelling App](chap-dhis2-connection.md), and includes mechanisms to disseminate your model to a large number of health ministries that use DHIS2 as their national health information system.

## How a model runs in Chap

A Chap model is a train script and a predict script, in R or Python, plus a short `MLproject` file that tells Chap how to run them. Once your model is Chap-compatible, you can use it in two ways:

- **On your own machine**, with the `chap` command-line tool. One command evaluates the model on a dataset and plots its forecasts:

    ```console
    chap eval --model-name . --dataset-csv data.csv --output-file eval.nc --plot --run-config.as-chapkit
    ```

- **In the Chap Modeling App**, where users can evaluate it on DHIS2 data, compare it with other models and make forecasts. For this, you [package the model as a Docker image](package_and_publish.md) and add it to a Chap installation, see [Running Your Own Model](../modeling-app/running-your-own-model.md).

## Where to start

First [set up your machine](setup.md). Then pick the guide that fits you:

| You... | Start here |
|---|---|
| already have a forecasting model in R or Python | [Bring your existing model to Chap](existing_model.md) |
| want to build a new model | [Build a new model from a template](mlproject_tutorial.md) |
| have a model that works with `chap eval` and want it in the Modeling App | [Package and publish your model](package_and_publish.md) |
| are not used to the terminal, git, or installing Python or R packages | [Development tools](development_tools.md), then come back here |
| want to learn statistical modelling and forecasting | [Learning modelling](learn_modelling.md) |

Both model guides use the same `MLproject` format and the same `chap eval` command. They only differ in where you start: from your own code, or from a template. The requirements a model has to meet are listed in [What Chap needs from your model](existing_model.md#1-what-chap-needs-from-your-model).

Strictly, Chap only requires that a model exposes the chapkit REST API. The train and predict scripts and the `MLproject` file are the recommended way to get there, since chapkit then provides the API for you. Advanced users can write a service that implements the API directly, see the [chapkit documentation](https://dhis2-chap.github.io/chapkit/). Most models do not need this.
