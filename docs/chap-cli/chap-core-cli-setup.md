# Setting up CHAP-Core CLI Tool

If you want to use CHAP Core on the commandline, develop custom models, or integrate external forecasting models with CHAP, you should install the `chap-core` Python package.

**Important: This guide is meant for end-users who need a stable version of CHAP Core.** If you are a developer and want to make changes or contribute to the CHAP Core codebase, you should follow the [getting started guide for contributors instead](../contributor/getting_started).

We recommend installing `chap-core` inside a Conda virtual environment. If you don't have Conda, you can install Miniconda (a minimal installer for Conda) from [Miniconda Installers](https://docs.anaconda.com/miniconda/#latest-miniconda-installer-links).

- **Windows**: After installation, open "Anaconda Prompt". Search for "Anaconda Prompt" in the Windows Start menu.
- **Linux**: Conda should work in your default terminal after installation.

Using Conda, you can create an environment like this:

```bash
$ conda create -n chap-core python=3.11
$ conda activate chap-core
```

After activating the environment, installing the latest version of `chap-core` can be done easily using pip: 

```bash
$ pip install chap-core
```

To install an older version of CHAP, for instance v1.0.1, you can instead do: 

```bash
$ pip install chap-core==1.0.1
```

To verify that the installation worked, check that you have the `chap` and `chap-cli` commands available in your terminal. 
For instance, typing `chap-cli` should give something like:

```console
Usage: chap-cli COMMAND

╭─ Commands ───────────────────────────────────────────────────────────────────────╮
│ evaluate   Evaluate how well a model would predict on the last year of the given │
│            dataset. Writes a report to the output file.                          │
│ harmonize  Harmonize health and population data from dhis2 with climate data     │
│            from google earth engine.                                             │
│ predict                                                                          │
│ --help,-h  Display this message and exit.                                        │
│ --version  Display application version.                                          │
╰──────────────────────────────────────────────────────────────────────────────────╯
```

Finally, depending on how you plan to use CHAP Core, you may need to [add Google Earth Engine credentials](earth-engine-auth) in order to retrieve climate data needed for running the models. 
