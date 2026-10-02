# Using the CLI Tool

The Chap CLI provides commands for evaluating disease prediction models, visualizing results, and exporting metrics.

## Quick Start

The main workflow consists of three commands:

1. **`chap eval`** - Run a backtest and export results to NetCDF format
2. **`chap plot-backtest`** - Generate visualizations from evaluation results
3. **`chap export-metrics`** - Export and compare metrics across evaluations

See the [Evaluation Workflow](evaluation-workflow.md) guide for detailed usage and examples.

Prefer a browser? [`chap ui`](ui.md) runs the same commands from a local web app: pick a dataset, start a model, run an evaluation and compare the results without typing a command.

## Documentation

- [Setup](chap-core-cli-setup.md) - How to install and configure the CLI
- [Evaluation Workflow](evaluation-workflow.md) - Complete guide to evaluating and comparing models
- [In the browser](ui.md) - `chap ui`, a local web app for the same workflow

## Command Reference

- [eval](eval-reference.md) - Full reference for the eval command with all parameters
- [report](report-reference.md) - Generate a PDF report from a trained model
- [explain-lime](explain-lime-reference.md) - Generate a LIME importance weighting for a single prediction
