# Contributor getting started

The main intended way of contributing to CHAP-Core is by contributing with models, for which we have a modularized system that makes it easy to contribute.
For this, we have guides/tutorials that [explain how to make models compatible with CHAP](../external_models/making_external_models_compatible).

We are also working on adding similar guides for contributing with custom code for evaluating models and visualizing results.
The code for evaluating and visualizing results is currently tightly integrated into the chap-core code base, but the plan is to
make this more modularized and easier to contribute to.

This document describes how to get started for contributing to the chap-core code base itself.

## Getting started working with the chap-core codebase

If you're new to CHAP Core, it can be useful to see [the code overview guide](code_overview) for a brief overview of the code base.

### Windows users

Windows users who wish to contribute to CHAP Core should [start by reading this important note](windows_contributors).

### Development setup

In order to make changes and contribute back to the chap-core Python codebase, you will need to [set up a development environment](./chap-contributor-setup.md).

Installing and activating the development environment above is a required step for the remaining steps below.

### Local installation of a DHIS2 instance with the DHIS2 Modeling app

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

### Code guidelines

In the current phase we are moving quite fast, and the code guidelines are not very strict.
However, we have some general guidelines that we try to follow:

- Alle code that is meant to be used should be tested (see the guidelines about testing)
- It is okay to have code that is not currently being used (just write a comment to explain)

### Debugging

Debugging can be done as usual in your favorite code editor.

For Windows users using VSCode, since the code should be run and tested on WSL, follow these steps to enable debugging in VSCode:

- Install the [WSL extension for WSL](https://code.visualstudio.com/docs/remote/wsl).
- Inside a wsl commandline session in your chap-core folder, type `code .`
- This will open your chap-core folder in VSCode using the WSL Linux/Python development environment. You can now use the VSCode debugger as usual.

### Testing

The CHAP Core codebase relies heavily on testing to ensure that the code works properly. A quick example to run a specific test file would be to write:

```bash
$ pytest tests/test_polygons.py
```

See more about our guidelines for testing in the [testing guide](testing).

### Code formatting

To ensure consistent and standardized code formatting we recommend running the `ruff` tool available from the development environment before making commits which will automatically check and report any formatting issues:

```bash
$ ruff check
```

### Docstring style guide

All docstrings should follow the [NumPy style guide](https://numpydoc.readthedocs.io/en/latest/format.html) for consistency and clarity.

Ensure that function and class docstrings include appropriate sections such as 'Parameters' and 'Returns'.

### Documentation

Changes to the CHAP Core documentation is done inside the `docs_source` folder, and can be built by writing:

```bash
$ cd docs_source
$ make html
```

More detailed guidelines for how to write and build the documentation [can be found here](writing_building_documentation.md).

### Contributing code

Code contributions should always be made to the `dev` branch first. When the `dev` branch has been used and tested for some time, the CHAP team will merge this into the `master` branch.

Before making your contribution, always [run the quick test suite](testing) to make sure everything works.

Most of the time, contributions should be made on a new branch, and creating a [Pull Request](https://github.com/dhis2-chap/chap-core/pulls) targeting the `dev` branch of the chap-core repository.

If you're an internal developer and only making small changes it's sometimes fine to push directly to the `dev` branch. However, for major changes or code refactoring, internal developers should still consider creating and submitting a PR for more systematic review of the code.
