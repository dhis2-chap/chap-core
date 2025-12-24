# Setting up CHAP Core CLI Tool

If you want to use CHAP Core on the command line, develop custom models, or integrate external forecasting models with CHAP, you should install the `chap-core` Python package.

**Important: This guide is for end-users who need a stable version of CHAP Core.** If you are a developer and want to make changes or contribute to the CHAP Core codebase, follow the [getting started guide for contributors](../contributor/getting_started.md) instead.

## Installation

We recommend using [uv](https://docs.astral.sh/uv/) for installation. If you don't have uv installed, you can install it with:

```console
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows, use:

```console
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then install `chap-core`:

```console
uv tool install chap-core
```

To install a specific version (e.g., v1.0.1):

```console
uv tool install chap-core==1.0.1
```

## Verify Installation

To verify that the installation worked, check that the `chap` command is available:

```bash
chap --help
```

You should see output listing available commands including `evaluate2`, `plot-backtest`, and `export-metrics`.

## Next Steps

- Follow the [Evaluation Workflow](evaluation-workflow.md) guide to evaluate and compare models
