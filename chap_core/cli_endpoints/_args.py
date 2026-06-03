"""Reusable cyclopts argument annotations for CHAP CLI commands.

Defining the ``Annotated[..., Parameter(...)]`` blocks once here keeps help
strings in sync across commands and avoids duplication.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from chap_core.api_types import BacktestParams, RunConfig

ModelNameArg = Annotated[
    str,
    Parameter(
        help="Model path (local directory), GitHub URL, or chapkit service URL. "
        "Examples: /path/to/model, https://github.com/org/model, http://localhost:8000"
    ),
]

DatasetCsvArg = Annotated[
    str,
    Parameter(
        help="Path or URL to CSV file containing disease data with columns: time_period, "
        "location, disease_cases, and climate covariates (rainfall, temperature, etc.)"
    ),
]

RunConfigArg = Annotated[
    RunConfig,
    Parameter(
        help="Model execution configuration. Use --run-config.is-chapkit-model for chapkit models, "
        "--run-config.debug for verbose logging, --run-config.ignore-environment to skip env setup"
    ),
]

BacktestParamsArg = Annotated[
    BacktestParams,
    Parameter(
        help="Backtest configuration. Use --backtest-params.n-periods for forecast horizon, "
        "--backtest-params.n-splits for number of train/test splits, "
        "--backtest-params.stride for step size between splits, "
        "--backtest-params.n-retrain for number of retrains across the splits"
    ),
]

ModelConfigYamlArg = Annotated[
    Path | None,
    Parameter(help="Path to YAML file with model-specific configuration parameters"),
]

DataSourceMappingArg = Annotated[
    Path | None,
    Parameter(
        help="Path to JSON file mapping model covariate names to CSV column names. "
        'Format: {"model_name": "csv_column"}. Example: {"rainfall": "precipitation_mm"}'
    ),
]
