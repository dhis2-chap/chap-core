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

README_TEMPLATE = """# {model_name}

A CHAP-compatible disease prediction model using linear regression.

## Setup

Install dependencies using uv:

```bash
uv sync
```

## Usage with CHAP

Evaluate the model using CHAP:

```bash
chap evaluate --model-name ./ --dataset-csv your_data.csv
```

## Manual usage

Train the model:

```bash
uv run python main.py train training_data.csv model.pkl
```

Generate predictions:

```bash
uv run python main.py predict model.pkl historic.csv future.csv predictions.csv
```

## Model description

This is a simple linear regression model that predicts disease cases based on:
- rainfall
- mean_temperature

You can modify `main.py` to use different features or a different model architecture.
"""


def init(model_name: str):
    """Initialize a new CHAP model project.

    Creates a new directory with the model name containing all necessary files
    for a CHAP-compatible model using uv for dependency management and cyclopts
    for argument parsing.

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

    # Write all template files
    (target_dir / "MLproject").write_text(MLPROJECT_TEMPLATE.format(model_name=model_name))
    (target_dir / "pyproject.toml").write_text(PYPROJECT_TEMPLATE.format(model_name=model_name))
    (target_dir / "main.py").write_text(MAIN_PY_TEMPLATE.format(model_name=model_name))
    (target_dir / "README.md").write_text(README_TEMPLATE.format(model_name=model_name))

    print(f"Created CHAP model project: {model_name}/")
    print("  - MLproject")
    print("  - pyproject.toml")
    print("  - main.py")
    print("  - README.md")
    print()
    print("Next steps:")
    print(f"  cd {model_name}")
    print("  uv sync")
    print("  chap evaluate --model-name ./ --dataset-csv your_data.csv")


def register_commands(app):
    """Register init commands with the CLI app."""
    app.command()(init)
