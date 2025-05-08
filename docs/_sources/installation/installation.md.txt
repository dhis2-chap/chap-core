# Installation and Getting Started

**Requirements:**

- CHAP has been tested to run on Windows, Linux, and MacOS. 
- CHAP can either be installed as a Python package, a commandline tool, a Docker container, or a DHIS2 App (see details below). 
- CHAP comes with a series of built-in forecasting models that are not very resource-intensive - 4 GB of RAM should be sufficient to run them. 

There are three main ways to interact with CHAP. Below we describe how to install CHAP for each of these use-cases:

## Option 1: Setting up Chap Modeling App in DHIS2

For most users we recommend interacting with Chap through the DHIS2 Modeling App interface, which serves as a frontend to the modeling platform.

* [Click here for instructions to setup the CHAP Modeling App in DHIS2](modeling-app-setup)

## Option 2: Setting up CHAP REST-API

Most users will also have to setup or connect to a REST-API that exposes the commandline tool to a url-endpoint. This is needed for applications that want to interact with CHAP over the internet, such as the Prediction App. 

There are two main ways to setup the CHAP REST-API:

- For local testing purposes, the easiest way is to [setup the CHAP REST-API locally with Docker](docker-compose-doc)
- For using CHAP in a production environment, it's recommended to instead follow the instructions for [setting up the CHAP REST-API on a server](running-chap-on-server)

## Option 3: Setting up CHAP Core CLI tool

If you want to interact with CHAP Core programmatically, you can do this through the CHAP Core commandline (CLI) tool. This lets you use CHAP Core on the commandline to do tasks like training and connecting to models, using the models to run predictions, or evaluating the results. 

* [Click here for instructions to install the CHAP Core CLI tool](chap-core-setup). 

> &#x1F6C8; **If you are a contributor** and would like to make changes to the CHAP Core codebase, you should instead follow [these instruction to setup CHAP Core in a development environment](chap-contributor-setup). 
