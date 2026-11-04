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

A Chap model is a train script and a predict script, in R or Python, plus a short `MLproject` file that tells Chap how to run them. You do not need to write any service or web code, and you do not need to install chapkit or run a Chap server. You evaluate the model on your own machine with one command:

```console
chap eval --model-name . --dataset-csv data.csv --output-file eval.nc --plot --run-config.as-chapkit
```

## Where to start

First [set up your machine](setup.md). Then pick the guide that fits you:

| You... | Start here |
|---|---|
| already have a forecasting model in R or Python | [Bring your existing model to Chap](existing_model.md) |
| want to build a new model | [Build a new model from a template](mlproject_tutorial.md) |
| are not used to the terminal, git, or installing Python or R packages | [Development tools](development_tools.md), then come back here |
| want to learn statistical modelling and forecasting | [Learning modelling](learn_modelling.md) |

Both model guides use the same `MLproject` format and the same `chap eval` command. They only differ in where you start: from your own code, or from a template.

Strictly, Chap only requires that a model exposes the chapkit REST API. The train and predict scripts and the `MLproject` file are the recommended way to get there, since chapkit then provides the API for you. Advanced users can write a service that implements the API directly, see the [chapkit documentation](https://dhis2-chap.github.io/chapkit/). Most models do not need this.
