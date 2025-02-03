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

In order to make changes and contribute back to the chap-core Python codebase, you will need to [set up a development environment](../installation/chap-contributor-setup.md). 

### CHAP Core versions

Most of the time, contributors should work with the latest code in the `dev` branch. However, in some cases it may be relevant to work with and test specific stable versions of CHAP Core. Versions are stored as git tags, so to see which versions of CHAP Core are available, you can write: 

```bash
git tag
```

Then you can switch to a desired version using, for instance: 

```bash
git switch tags/v1.0.3
```

### Testing

The CHAP Core codebase relies heavily on testing to ensure that the code works properly. See more about our guidelines for testing in the [testing guide](testing). 

### Contributing code

Code contributions should mainly happen by creating a pull request in the chap-core repository. In order to do this, you
will have to have a clone of the chap-core repository on github (which is possible for anyone with a github account).

Some internal developers can also push directly to the main chap-core repository.
