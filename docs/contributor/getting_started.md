# Contributor getting started

The main intended way of contributing to Chap Core is by contributing with models, for which we have a modularized system that makes it easy to contribute.
For this, we have guides/tutorials that explain how to make models compatible with Chap.

We are also working on adding similar guides for contributing with custom code for evaluating models and visualizing results.
The code for evaluating and visualizing results is currently tightly integrated into the chap-core code base, but the plan is to
make this more modularized and easier to contribute to.

This document describes how to get started for contributing to the chap-core code base itself.

## Getting started working with the chap-core codebase

If you're new to Chap Core, it can be useful to see [the code overview guide](code_overview.md) for a brief overview of the code base.

### Windows users

Windows users who wish to contribute to Chap Core should [start by reading this important note](windows_contributors.md).

### Development setup

In order to make changes and contribute back to the chap-core Python codebase, you will need to [set up a development environment](./chap-contributor-setup.md).

Installing and activating the development environment above is a required step for the remaining steps below.

### Local installation of a DHIS2 instance with the DHIS2 Modeling app

If you want to test chap-core with the Modeling app, see [Local installation of a DHIS2 instance with the DHIS2 Modeling app](local-dhis2-installation.md) for how to set up a local installation of DHIS2.

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

The Chap Core codebase relies heavily on testing to ensure that the code works properly. A quick example to run a specific test file would be to write:

```bash
$ pytest tests/test_polygons.py
```

See more about our guidelines for testing in the [testing guide](testing.md).

### Code formatting

Before making commits, run the linters and formatters, and then the same checks that CI runs:

```bash
$ make lint   # fixes what ruff can fix
$ make check  # what CI runs: ruff, mypy, pyright and alembic head check
```

### Docstring style guide

All docstrings should follow the [NumPy style guide](https://numpydoc.readthedocs.io/en/latest/format.html) for consistency and clarity.

Ensure that function and class docstrings include appropriate sections such as 'Parameters' and 'Returns'.

### Documentation

Changes to the Chap Core documentation are done inside the `docs` folder. Build the site from the project root with:

```bash
$ make docs
```

More detailed guidelines for how to write and build the documentation [can be found here](writing_building_documentation.md).

### Contributing code

Code contributions are made on a new branch and submitted as a [Pull Request](https://github.com/dhis2-chap/chap-core/pulls) targeting the `master` branch of the chap-core repository.

Before making your contribution, always [run the quick test suite](testing.md) to make sure everything works.

Fill in the PR template, including the `Review needed:` line, which tells reviewers how much scrutiny the change needs. CI fails PRs that leave it out. See [Pull Request Review](pr_review.md) for the options and how reviews work.
