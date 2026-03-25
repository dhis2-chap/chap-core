# Generated Features

Chap can automatically compute derived features for your model, such as clustering locations by their seasonal disease patterns. These features are generated at runtime from the input data, so they do not need to be provided as part of the dataset.

## How It Works

You request generated features by adding entries with a `gen:` prefix to the `required_covariates` list in your MLproject file. For example:

```yaml
name: my_model

required_covariates:
  - rainfall
  - mean_temperature
  - gen:seasonality_cluster

docker_env:
  image: python:3.13

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

When Chap runs this model:

1. The regular covariates (`rainfall`, `mean_temperature`) are expected in the input dataset as usual
2. Generated features (`gen:seasonality_cluster`) are computed by Chap and added to the data before your model receives it
3. Your model's training and prediction data will include a `cluster_id` column alongside the regular covariates

## Available Generated Features

| Config value | Column added | Description |
|---|---|---|
| `gen:seasonality_cluster` | `cluster_id` | Assigns each location to a cluster (0, 1, 2, ...) based on its normalized seasonal disease profile, using KMeans clustering |

## How Generated Features Appear in Your Data

Generated features are added as regular columns to the CSV files your model receives. For `gen:seasonality_cluster`, the training data CSV will look like:

```text
time_period,location,disease_cases,rainfall,mean_temperature,cluster_id
2023-W01,location_A,42,120.5,25.3,0.0
2023-W01,location_B,38,95.2,22.1,1.0
2023-W02,location_A,45,110.0,24.8,0.0
2023-W02,location_B,41,100.1,21.5,1.0
```

The `cluster_id` value is constant for each location (it does not change over time). During prediction, Chap copies the cluster assignments from the historic data to the future data so that your model has access to them in both phases.

## Validation

Chap validates `gen:` entries at configuration time:

- Regular covariates are checked against the dataset columns as usual
- `gen:` covariates are not expected in the input data and will not trigger a "missing covariate" error
- If a `gen:` entry references an unknown generator, Chap reports an error

## When to Use Generated Features

Generated features are useful when your model benefits from information that is derived from the full dataset rather than provided as raw input. For example:

- **Seasonality clustering** groups locations with similar disease patterns, allowing your model to share information across similar regions
- **Derived indices** that combine multiple raw covariates into a single feature

Because the generation happens inside Chap, you do not need to precompute these features yourself or include them in your input data files.
