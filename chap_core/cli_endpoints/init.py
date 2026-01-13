"""CLI endpoint for initializing new CHAP model projects."""

import logging
import subprocess
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

MLPROJECT_TEMPLATE = """name: {model_name}

uv_env: pyproject.toml

entry_points:
  train:
    parameters:
      train_data: str
      model: str
    command: "python main.py train {{train_data}} {{model}}"
  predict:
    parameters:
      historic_data: str
      future_data: str
      model: str
      out_file: str
    command: "python main.py predict {{model}} {{historic_data}} {{future_data}} {{out_file}}"
"""

PYPROJECT_TEMPLATE = """[project]
name = "{model_name}"
version = "0.1.0"
description = "A CHAP-compatible disease prediction model"
requires-python = ">=3.10"
dependencies = [
    "pandas",
    "scikit-learn",
    "joblib",
    "cyclopts",
]
"""

MAIN_PY_TEMPLATE = '''"""CHAP model: {model_name}

A simple linear regression model for disease prediction.
"""

import joblib
import pandas as pd
from cyclopts import App
from sklearn.linear_model import LinearRegression

app = App()


@app.command()
def train(train_data: str, model: str):
    """Train the model on the provided data.

    Parameters
    ----------
    train_data
        Path to the training data CSV file.
    model
        Path where the trained model will be saved.
    """
    df = pd.read_csv(train_data)
    features = df[["rainfall", "mean_temperature"]].fillna(0)
    target = df["disease_cases"].fillna(0)

    reg = LinearRegression()
    reg.fit(features, target)
    joblib.dump(reg, model)
    print(f"Model saved to {{model}}")


@app.command()
def predict(model: str, historic_data: str, future_data: str, out_file: str):
    """Generate predictions using the trained model.

    Parameters
    ----------
    model
        Path to the trained model file.
    historic_data
        Path to historic data CSV file (unused in this simple model).
    future_data
        Path to future climate data CSV file.
    out_file
        Path where predictions will be saved.
    """
    reg = joblib.load(model)
    future_df = pd.read_csv(future_data)
    features = future_df[["rainfall", "mean_temperature"]].fillna(0)

    predictions = reg.predict(features)
    output_df = future_df[["time_period", "location"]].copy()
    output_df["sample_0"] = predictions
    output_df.to_csv(out_file, index=False)
    print(f"Predictions saved to {{out_file}}")


if __name__ == "__main__":
    app()
'''

ISOLATED_RUN_TEMPLATE = '''"""Run the model in isolation without CHAP integration.

This script demonstrates how to train and predict using the model directly,
which is useful for development and debugging before integrating with CHAP.
"""

import subprocess

# Train the model
subprocess.run(
    ["uv", "run", "python", "main.py", "train", "input/trainData.csv", "output/model.pkl"],
    check=True,
)

# Generate predictions
subprocess.run(
    [
        "uv",
        "run",
        "python",
        "main.py",
        "predict",
        "output/model.pkl",
        "input/trainData.csv",
        "input/futureClimateData.csv",
        "output/predictions.csv",
    ],
    check=True,
)

print("\\nPredictions saved to output/predictions.csv")
'''

TRAIN_DATA_TEMPLATE = """time_period,rainfall,mean_temperature,disease_cases,location
2023-05,10,30,200,loc1
2023-06,2,30,100,loc1
"""

FUTURE_DATA_TEMPLATE = """time_period,rainfall,mean_temperature,location
2023-07,20,20,loc1
2023-08,30,20,loc1
2023-09,30,30,loc1
"""

README_TEMPLATE = """# {model_name}

This is a CHAP-compatible forecasting model using [uv](https://docs.astral.sh/uv/) for dependency management and [cyclopts](https://cyclopts.readthedocs.io/) for command-line argument parsing.

The model learns a linear regression from rainfall and temperature to disease cases in the same month. It is meant as a starting point for developing your own model.

## Requirements

Make sure you have [uv](https://docs.astral.sh/uv/) installed. No other setup is needed - `uv run` will automatically create the virtual environment and install dependencies on first use.

## Running the model without CHAP integration

Before integrating with CHAP, you can test the model directly using the included sample data:

```bash
uv run python isolated_run.py
```

Or run the commands manually:

### Training the model

```bash
uv run python main.py train input/trainData.csv output/model.pkl
```

### Generating predictions

```bash
uv run python main.py predict output/model.pkl input/trainData.csv input/futureClimateData.csv output/predictions.csv
```

## Running the model with CHAP

After installing chap-core, run:

```bash
chap evaluate --model-name /path/to/{model_name} --dataset-csv your_data.csv --report-filename report.pdf
```

Or with a built-in dataset:

```bash
chap evaluate --model-name /path/to/{model_name} --dataset-name ISIMIP_dengue_harmonized --dataset-country brazil --report-filename report.pdf
```

## Project structure

- `MLproject` - Defines how CHAP interacts with your model
- `pyproject.toml` - Lists your Python dependencies
- `main.py` - Contains your model's train and predict logic
- `isolated_run.py` - Script to test the model without CHAP
- `input/` - Sample training and future climate data
- `output/` - Where trained models and predictions are saved

## Customizing the model

Edit `main.py` to change:
- Which features are used for prediction
- The machine learning algorithm (currently LinearRegression)
- How predictions are formatted

Add dependencies to `pyproject.toml` - they will be installed automatically on the next `uv run`.
"""

# R/renv templates
R_MLPROJECT_TEMPLATE = """name: {model_name}

renv_env: renv.lock

entry_points:
  train:
    parameters:
      train_data: str
      model: str
    command: "Rscript main.R train --train_data {{train_data}} --model {{model}}"
  predict:
    parameters:
      historic_data: str
      future_data: str
      model: str
      out_file: str
    command: "Rscript main.R predict --model {{model}} --historic_data {{historic_data}} --future_data {{future_data}} --out_file {{out_file}}"
"""

R_MAIN_TEMPLATE = """# CHAP model: {model_name}
#
# A simple linear regression model for disease prediction.

library(optparse)

train <- function(train_data_path, model_path) {{
  df <- read.csv(train_data_path)

  df$rainfall[is.na(df$rainfall)] <- 0
  df$mean_temperature[is.na(df$mean_temperature)] <- 0
  df$disease_cases[is.na(df$disease_cases)] <- 0

  model <- lm(disease_cases ~ rainfall + mean_temperature, data = df)
  saveRDS(model, model_path)
  message(paste("Model saved to", model_path))
}}

predict_model <- function(model_path, historic_data_path, future_data_path, out_path) {{
  model <- readRDS(model_path)
  future_df <- read.csv(future_data_path)

  future_df$rainfall[is.na(future_df$rainfall)] <- 0
  future_df$mean_temperature[is.na(future_df$mean_temperature)] <- 0

  predictions <- predict(model, newdata = future_df)

  output_df <- data.frame(
    time_period = future_df$time_period,
    location = future_df$location,
    sample_0 = predictions
  )
  write.csv(output_df, out_path, row.names = FALSE)
  message(paste("Predictions saved to", out_path))
}}

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {{
  stop("Usage: Rscript main.R <train|predict> [options]")
}}

command <- args[1]
command_args <- args[-1]

if (command == "train") {{
  option_list <- list(
    make_option(c("-t", "--train_data"), type = "character", help = "Path to training data CSV"),
    make_option(c("-m", "--model"), type = "character", help = "Path to save trained model")
  )
  parser <- OptionParser(option_list = option_list, usage = "usage: %prog train [options]")
  opts <- parse_args(parser, args = command_args, positional_arguments = TRUE)

  if (is.null(opts$options$train_data) || is.null(opts$options$model)) {{
    print_help(parser)
    stop("train requires --train_data and --model")
  }}

  train(opts$options$train_data, opts$options$model)

}} else if (command == "predict") {{
  option_list <- list(
    make_option(c("-m", "--model"), type = "character", help = "Path to trained model"),
    make_option(c("-d", "--historic_data"), type = "character", help = "Path to historic data CSV"),
    make_option(c("-f", "--future_data"), type = "character", help = "Path to future climate data CSV"),
    make_option(c("-o", "--out_file"), type = "character", help = "Path to save predictions")
  )
  parser <- OptionParser(option_list = option_list, usage = "usage: %prog predict [options]")
  opts <- parse_args(parser, args = command_args, positional_arguments = TRUE)

  if (is.null(opts$options$model) || is.null(opts$options$historic_data) ||
      is.null(opts$options$future_data) || is.null(opts$options$out_file)) {{
    print_help(parser)
    stop("predict requires --model, --historic_data, --future_data, and --out_file")
  }}

  predict_model(opts$options$model, opts$options$historic_data,
                opts$options$future_data, opts$options$out_file)

}} else {{
  stop(paste("Unknown command:", command, "\\nUsage: Rscript main.R <train|predict> [options]"))
}}
"""

R_ISOLATED_RUN_TEMPLATE = """# Run the model in isolation without CHAP integration.
#
# This script demonstrates how to train and predict using the model directly,
# which is useful for development and debugging before integrating with CHAP.

# Restore renv packages if needed
renv::restore(prompt = FALSE)

# Train the model
system2("Rscript", c("main.R", "train",
                     "--train_data", "input/trainData.csv",
                     "--model", "output/model.rds"))

# Generate predictions
system2("Rscript", c("main.R", "predict",
                     "--model", "output/model.rds",
                     "--historic_data", "input/trainData.csv",
                     "--future_data", "input/futureClimateData.csv",
                     "--out_file", "output/predictions.csv"))

message("\\nPredictions saved to output/predictions.csv")
"""

R_README_TEMPLATE = """# {model_name}

This is a CHAP-compatible forecasting model using [renv](https://rstudio.github.io/renv/) for dependency management.

The model learns a linear regression from rainfall and temperature to disease cases in the same month. It is meant as a starting point for developing your own model.

## Requirements

- R (>= 4.0)
- renv will be automatically restored on first run

## Running the model without CHAP integration

Before integrating with CHAP, you can test the model directly using the included sample data:

```bash
Rscript isolated_run.R
```

Or run the commands manually:

### Training the model

```bash
Rscript main.R train --train_data input/trainData.csv --model output/model.rds
```

### Generating predictions

```bash
Rscript main.R predict --model output/model.rds --historic_data input/trainData.csv --future_data input/futureClimateData.csv --out_file output/predictions.csv
```

## Running the model with CHAP

After installing chap-core, run:

```bash
chap evaluate --model-name /path/to/{model_name} --dataset-csv your_data.csv --report-filename report.pdf
```

Or with a built-in dataset:

```bash
chap evaluate --model-name /path/to/{model_name} --dataset-name ISIMIP_dengue_harmonized --dataset-country brazil --report-filename report.pdf
```

## Project structure

- `MLproject` - Defines how CHAP interacts with your model
- `renv.lock` - Locks R package versions
- `main.R` - Contains your model's train and predict logic
- `isolated_run.R` - Script to test the model without CHAP
- `input/` - Sample training and future climate data
- `output/` - Where trained models and predictions are saved

## Customizing the model

Edit `main.R` to change:
- Which features are used for prediction
- The machine learning algorithm (currently lm)
- How predictions are formatted

Add dependencies using `renv::install("package_name")` and then `renv::snapshot()` to update renv.lock.
"""


def _check_r_and_renv_available():
    """Check if R and renv are available."""
    # Check if R is available
    try:
        subprocess.run(
            ["Rscript", "--version"],
            capture_output=True,
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError("R is not installed or not in PATH. Please install R first.")

    # Check if renv is available
    result = subprocess.run(
        ["Rscript", "-e", "if (!requireNamespace('renv', quietly=TRUE)) quit(status=1)"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "renv package is not installed. Please install it with: Rscript -e \"install.packages('renv')\""
        )


def _init_renv(target_dir: Path):
    """Initialize renv in the target directory by running renv commands."""
    _check_r_and_renv_available()

    # Initialize renv in the directory (creates renv/, .Rprofile, renv.lock)
    subprocess.run(
        ["Rscript", "-e", "renv::init()"],
        cwd=target_dir,
        check=True,
    )

    # Install required packages (optparse for CLI argument parsing)
    subprocess.run(
        ["Rscript", "-e", "renv::install('optparse')"],
        cwd=target_dir,
        check=True,
    )

    # Snapshot to update renv.lock with the installed packages
    subprocess.run(
        ["Rscript", "-e", "renv::snapshot(prompt=FALSE)"],
        cwd=target_dir,
        check=True,
    )


def init(model_name: str, language: Literal["python", "r"] = "python"):
    """Initialize a new CHAP model project.

    Creates a new directory with the model name containing all necessary files
    for a CHAP-compatible model. Includes sample data for testing.

    Parameters
    ----------
    model_name
        Name of the model project to create. A directory with this name will be
        created in the current working directory.
    language
        Programming language for the model: "python" (default) or "r".
    """
    target_dir = Path.cwd() / model_name

    if target_dir.exists():
        raise FileExistsError(f"Directory '{model_name}' already exists")

    target_dir.mkdir(parents=True)
    (target_dir / "input").mkdir()
    (target_dir / "output").mkdir()

    # Common files
    (target_dir / "input" / "trainData.csv").write_text(TRAIN_DATA_TEMPLATE)
    (target_dir / "input" / "futureClimateData.csv").write_text(FUTURE_DATA_TEMPLATE)
    (target_dir / "output" / ".gitkeep").write_text("")

    if language == "python":
        # Write Python template files
        (target_dir / "MLproject").write_text(MLPROJECT_TEMPLATE.format(model_name=model_name))
        (target_dir / "pyproject.toml").write_text(PYPROJECT_TEMPLATE.format(model_name=model_name))
        (target_dir / "main.py").write_text(MAIN_PY_TEMPLATE.format(model_name=model_name))
        (target_dir / "README.md").write_text(README_TEMPLATE.format(model_name=model_name))
        (target_dir / "isolated_run.py").write_text(ISOLATED_RUN_TEMPLATE)

        print(f"Created CHAP model project: {model_name}/")
        print("  - MLproject")
        print("  - pyproject.toml")
        print("  - main.py")
        print("  - README.md")
        print("  - isolated_run.py")
        print("  - input/trainData.csv")
        print("  - input/futureClimateData.csv")
        print("  - output/")
        print()
        print("Next steps:")
        print(f"  cd {model_name}")
        print("  uv run python isolated_run.py  # Test the model")
        print("  chap evaluate --model-name ./ --dataset-csv your_data.csv")
    else:
        # Write R template files
        (target_dir / "MLproject").write_text(R_MLPROJECT_TEMPLATE.format(model_name=model_name))
        (target_dir / "main.R").write_text(R_MAIN_TEMPLATE.format(model_name=model_name))
        (target_dir / "README.md").write_text(R_README_TEMPLATE.format(model_name=model_name))
        (target_dir / "isolated_run.R").write_text(R_ISOLATED_RUN_TEMPLATE)
        # renv::init() creates .Rprofile, renv.lock, and renv/ directory
        _init_renv(target_dir)

        print(f"Created CHAP model project: {model_name}/")
        print("  - MLproject")
        print("  - main.R")
        print("  - README.md")
        print("  - isolated_run.R")
        print("  - .Rprofile")
        print("  - renv.lock")
        print("  - renv/activate.R")
        print("  - input/trainData.csv")
        print("  - input/futureClimateData.csv")
        print("  - output/")
        print()
        print("Next steps:")
        print(f"  cd {model_name}")
        print("  Rscript isolated_run.R  # Test the model")
        print("  chap evaluate --model-name ./ --dataset-csv your_data.csv")


def register_commands(app):
    """Register init commands with the CLI app."""
    app.command()(init)
