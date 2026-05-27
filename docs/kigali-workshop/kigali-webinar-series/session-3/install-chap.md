# 1. Installing Chap
In this guide, you'll install the Chap command-line tool. Once installed, you can run chap eval to test any model against real datasets — which you'll do in the next guide in this session.

## Why Chap?
Chap (Climate and Health Assessment Platform) is a tool for developing and evaluating disease prediction models that use climate data. The `chap` command-line tool allows you to:

- Evaluate models on historical data
- Compare model performance
- Test your models before integrating them with DHIS2

## Prerequisites

You should have `uv` installed from [Session 2](../session-2/virtual-environments.md). If not, install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows users:** Use WSL (Windows Subsystem for Linux) as covered in [Session 2](../session-2/terminal.md).

## Installing Chap

Install Chap as a global tool using uv:

```bash
uv tool install chap-core --python 3.13
```

This installs the `chap` command-line tool globally, making it available from any directory.

## Exercise

### Verify your installation

Run the following command:

```bash
chap --help
```

You should see output listing available commands including `eval`, `plot-backtest`, and `export-metrics`.

**Verification:** If you see the help output with available commands, Chap is installed correctly. You're ready for the next guide: [Implement your own model from a minimalist example](fork-example.md).
