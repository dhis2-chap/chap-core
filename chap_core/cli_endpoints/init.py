"""CLI endpoint for initializing new CHAP model projects."""

import logging
from pathlib import Path

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

## Setting Up the Environment

Make sure you have [uv](https://docs.astral.sh/uv/) installed, then run:

```bash
uv sync
```

This will create a virtual environment and install all dependencies automatically.

## Running the model without CHAP integration

Before integrating with CHAP, you can test the model directly using the included sample data:

```bash
python isolated_run.py
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
- `input/` - Sample training and future climate data
- `output/` - Where trained models and predictions are saved

## Customizing the model

Edit `main.py` to change:
- Which features are used for prediction
- The machine learning algorithm (currently LinearRegression)
- How predictions are formatted

Add dependencies to `pyproject.toml` and run `uv sync` to install them.
"""


def init(model_name: str):
    """Initialize a new CHAP model project.

    Creates a new directory with the model name containing all necessary files
    for a CHAP-compatible model using uv for dependency management and cyclopts
    for argument parsing. Includes sample data for testing.

    Parameters
    ----------
    model_name
        Name of the model project to create. A directory with this name will be
        created in the current working directory.
    """
    target_dir = Path.cwd() / model_name

    if target_dir.exists():
        raise FileExistsError(f"Directory '{model_name}' already exists")

    target_dir.mkdir(parents=True)
    (target_dir / "input").mkdir()
    (target_dir / "output").mkdir()

    # Write all template files
    (target_dir / "MLproject").write_text(MLPROJECT_TEMPLATE.format(model_name=model_name))
    (target_dir / "pyproject.toml").write_text(PYPROJECT_TEMPLATE.format(model_name=model_name))
    (target_dir / "main.py").write_text(MAIN_PY_TEMPLATE.format(model_name=model_name))
    (target_dir / "README.md").write_text(README_TEMPLATE.format(model_name=model_name))
    (target_dir / "isolated_run.py").write_text(ISOLATED_RUN_TEMPLATE)
    (target_dir / "input" / "trainData.csv").write_text(TRAIN_DATA_TEMPLATE)
    (target_dir / "input" / "futureClimateData.csv").write_text(FUTURE_DATA_TEMPLATE)
    (target_dir / "output" / ".gitkeep").write_text("")

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
    print("  uv sync")
    print("  python isolated_run.py  # Test the model")
    print("  chap evaluate --model-name ./ --dataset-csv your_data.csv")


def register_commands(app):
    """Register init commands with the CLI app."""
    app.command()(init)
