# Describing your model in our yaml-based format

To make your model chap-compatible, you need your train and predict endpoints (as discussed [here](train_and_predict.md)) need to be formally defined in a [YAML format](https://en.wikipedia.org/wiki/YAML) that follows the popular [MLflow standard](https://www.mlflow.org/docs/latest/projects.html#project-format).
Your codebase need to contain a file named `MLproject`. After adding the MLproject-file, you project may look like this:

```
my_model/
├── MLproject
├── input (input data, e.g. disease, climate)
├── output (predcitions / evaluations)
├── train.py (or R-file)
├── pyproject.toml (or R-file)
└── predict.py (or R-file)
```

## MLproject
- An entry point in the MLproject file called `train` with parameters `train_data` and `model`
- An entry point in the MLproject file called `predict` with parameters `historic_data`, `future_data`, `model` and `out_file`

These should contain commands that can be run to train a model and predict the future using that model. The model parameter should be used to save a model in the train step that can be read and used in the predict step. Chap will provide all the data (the other parameters) when running a model.

### Example MLproject file

Taken from our minimalist_example

```yaml
name: minimalist_example_uv

uv_env: pyproject.toml

entry_points:
  train:
    parameters:
      train_data: str
      model: str
    command: "python train.py {train_data} {model}"
  predict:
    parameters:
      historic_data: str
      future_data: str
      model: str
      out_file: str
    command: "python predict.py {model} {historic_data} {future_data} {out_file}"
```

### Predict and train

When Chap runs your model, it calls the `train` and `predict` commands defined in your MLproject file. Chap creates CSV files with the data and passes their filenames as command-line arguments by substituting the `{parameter}` placeholders in the command string.

#### Train

Chap calls the `train` entry point with two parameters:

| Parameter | Description |
|-----------|-------------|
| `train_data` | Filename of a CSV file containing the training data |
| `model` | Filename where your script must save the trained model |

Your train script should:

1. Read the CSV from `train_data` (columns: `time_period`, `location`, `disease_cases`, plus any covariates)
2. Fit your model on this data
3. Save the trained model to the path given by `model` (format is up to you: JSON, pickle, RDS, etc.)

**Example train script (Python):**

```python
import json
import sys
import pandas as pd

def train(training_data_filename: str, model_path: str):
    df = pd.read_csv(training_data_filename)
    stats = df.groupby("location")["disease_cases"].agg(["mean", "std"]).to_dict()
    with open(model_path, "w") as f:
        json.dump(stats, f)
```

Call the function from the command line:

```console
# train.py (bottom of file)
if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])
```

#### Predict

Chap calls the `predict` entry point with four parameters:

| Parameter | Description |
|-----------|-------------|
| `historic_data` | Filename of a CSV with observed data up to the prediction period |
| `future_data` | Filename of a CSV with the future time periods and covariates to predict for |
| `model` | Filename of the saved model (produced by the train step) |
| `out_file` | Filename where your script must write the predictions |

Your predict script should:

1. Load the trained model from `model`
2. Read `historic_data` and/or `future_data` as needed
3. Generate predictions for each `(location, time_period)` row in `future_data`
4. Write a CSV to `out_file` with columns: `time_period`, `location`, `sample_0`, `sample_1`, ..., `sample_N`

Each `sample_i` column represents one draw from the predictive distribution. Chap uses these samples to compute uncertainty intervals. Typically, models produce 100 samples.

**Example predict script (Python):**

```python
import json
import sys
import numpy as np
import pandas as pd

def predict(model_filename: str, historic_data_filename: str,
            future_data_filename: str, output_filename: str):
    with open(model_filename) as f:
        stats = json.load(f)

    future_df = pd.read_csv(future_data_filename)
    n_samples = 100

    rows = []
    for _, row in future_df.iterrows():
        loc = row["location"]
        mean = stats["mean"].get(loc, 0)
        std = stats["std"].get(loc, 1) or 1
        samples = np.maximum(0, np.random.normal(mean, std, n_samples))
        row_data = {"time_period": row["time_period"], "location": loc}
        row_data.update({f"sample_{i}": s for i, s in enumerate(samples)})
        rows.append(row_data)

    pd.DataFrame(rows).to_csv(output_filename, index=False)
```

Call the function from the command line:

```console
# predict.py (bottom of file)
if __name__ == "__main__":
    predict(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
```

#### How parameters map to files

When Chap executes the command `"python train.py {train_data} {model}"`, it replaces `{train_data}` and `{model}` with the actual filenames of the CSV and model files it has created. Both train and predict run in the same working directory, so the model file saved during training is directly accessible during prediction.

For example, given the MLproject entry:

```yaml
command: "python predict.py {model} {historic_data} {future_data} {out_file}"
```

Chap might execute:

```console
python predict.py model.json historic_data_2024-01-01.csv future_data_2024-01-01.csv predictions_2024-01-01.csv
```

#### Optional parameters

Your MLproject entry points can also accept these optional parameters:

| Parameter | Description |
|-----------|-------------|
| `polygons` | Filename of a GeoJSON file with location polygon boundaries (only passed if spatial data is available) |
| `model_config` | Filename of a YAML configuration file for model-specific settings |

To use these, include the corresponding `{polygons}` or `{model_config}` placeholder in your command string

### ML file environment

The MLproject file can specify a docker image, Python virtual environment, uv-managed environment, or renv environment (for R models) that will be used when running the commands.
An example of this is the [MLproject file](https://github.com/dhis2-chap/minimalist_example_r/blob/main/MLproject) contained within our minimalist_example_r.

## Next steps

- For Python-specific environment setup, see [Python-models](python_models.md)
- For R-specific environment setup, see [R-models](r_models.md)
