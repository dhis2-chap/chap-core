# Installation and Getting Started

## Requirements

The `chap-core` Python package requires Python 3.10 or higher and is tested to work on Windows, Linux, and macOS. 

Most of the built-in forecasting models in CHAP require Docker to run, and we recommend also using Docker to set up 
`chap-core` for most use cases (see below). The current built-in models are not resource-intensive, and one should be 
fine with 4 GB of RAM.

Installation of CHAP depends on how you want to use it. Here we present several possible ways to use and install CHAP:

## Setting up CHAP Prediction App in DHIS2

For most users we recommend interacting with CHAP through the DHIS2 Prediction App interface.

[Click here for instructions to setup the CHAP Prediction App in DHIS2](prediction-app-setup)

## Setting up CHAP REST-API

Most users will also have to setup or connect to a REST-API that exposes the commandline tool to a url-endpoint. This is needed for applications that want to interact with CHAP over the internet, such as the Prediction App. 

There are two main ways to setup the CHAP REST-API:

- For local testing purposes, the easiest way is to [setup the CHAP REST-API locally with Docker](docker-compose-doc)
- For using CHAP in a production environment, it's recommended to instead follow the instructions for [setting up the CHAP REST-API on a server](running-chap-on-server)

## Setting up CHAP Core CLI tool

If you are an advanced user you will most likely want the CHAP Core commandline (CLI) tool. This lets you interact programmatically with CHAP Core on the commandline to do tasks like training and connecting to models, using the models to run predictions, or evaluating the results. 

[Click here for instructions to setup the CHAP Core CLI tool](chap-core-setup)
