# Using the CLI Tool

The Chap CLI provides commands for evaluating disease prediction models, visualizing results, and exporting metrics.

## Quick Start

The main workflow consists of three commands:

1. **`chap eval`** - Run a backtest and export results to NetCDF format
2. **`chap plot-backtest`** - Generate visualizations from evaluation results
3. **`chap export-metrics`** - Export and compare metrics across evaluations

See the [Evaluation Workflow](evaluation-workflow.md) guide for detailed usage and examples.

To set up a model and dataset once and then run `chap eval` without options, create a [Chap project](projects.md) with `chap init`.

## Documentation

- [Setup](chap-core-cli-setup.md) - How to install and configure the CLI
- [Evaluation Workflow](evaluation-workflow.md) - Complete guide to evaluating and comparing models
- [Chap Projects](projects.md) - Keep a model, dataset and settings in a directory and run `chap eval` without options

## Command Reference

- [eval](eval-reference.md) - Full reference for the eval command with all parameters
- [report](report-reference.md) - Generate a PDF report from a trained model
- [explain-lime](explain-lime-reference.md) - Generate a LIME importance weighting for a single prediction
- [simulate](simulate-reference.md) - Generate a synthetic dataset from a simulation model
