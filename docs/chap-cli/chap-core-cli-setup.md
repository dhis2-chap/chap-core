# Setting up CHAP Core CLI Tool

If you want to use CHAP Core on the command line, develop custom models, or integrate external forecasting models with CHAP, you should install the `chap-core` Python package.

**Important: This guide is for end-users who need a stable version of CHAP Core.** If you are a developer and want to make changes or contribute to the CHAP Core codebase, follow the [getting started guide for contributors](../contributor/getting_started.md) instead.

## Installation

We recommend installing `chap-core` inside a Conda virtual environment. If you don't have Conda, you can install Miniconda (a minimal installer for Conda) from [Miniconda Installers](https://docs.anaconda.com/miniconda/#latest-miniconda-installer-links).

- **Windows**: After installation, open "Anaconda Prompt". Search for "Anaconda Prompt" in the Windows Start menu.
- **Linux/macOS**: Conda should work in your default terminal after installation.

Create and activate a Conda environment:

```console
conda create -n chap-core python=3.11
conda activate chap-core
```

Install the latest version of `chap-core` using pip:

```console
pip install chap-core
```

To install a specific version (e.g., v1.0.1):

```console
pip install chap-core==1.0.1
```

## Verify Installation

To verify that the installation worked, check that the `chap` command is available:

```bash
chap --help
```

You should see output listing available commands including `evaluate2`, `plot-backtest`, and `export-metrics`.

## Next Steps

- Follow the [Evaluation Workflow](evaluation-workflow.md) guide to evaluate and compare models
