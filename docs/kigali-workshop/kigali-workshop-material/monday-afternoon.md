# Monday Afternoon - 23 Feb

See the [Evaluation Walkthrough](../../contributor/evaluation_walkthrough.md) for a
hands-on, step-by-step guide through the evaluation pipeline.

## Workshop: Evaluating Models on the Laos Dengue Dataset

In this hands-on exercise you will download a real dataset, run two different models through the CHAP evaluation pipeline, and compare the results.

### 1. Download the dataset

Create a working directory and download the Laos dengue dataset (monthly, admin-1 level):

```console
$ mkdir laos-workshop && cd laos-workshop
$ curl -sL -o chap_LAO_admin1_monthly.csv \
    "https://raw.githubusercontent.com/dhis2/climate-health-data/main/lao/chap_LAO_admin1_monthly.csv"
```

The CSV contains ~2800 rows covering 18 provinces from 1998-2010 with columns: `time_period`, `location`, `disease_cases`, `population`, `location_name`, `rainfall`, `mean_temperature`, `mean_relative_humidity`.

### 2. Explore the dataset

```console
$ chap plot-dataset chap_LAO_admin1_monthly.csv
```

This opens an interactive plot in your browser showing standardized disease cases and climate features across all locations:

![Laos dataset overview](assets/laos_dataset.png)

### 3. Evaluate Model A -- Minimalist baseline

Run a backtest with 2 train/test splits and a 3-month forecast horizon using a simple baseline model:

```console
$ chap eval https://github.com/knutdrand/minimalist_example_uv \
    chap_LAO_admin1_monthly.csv \
    minimalist_eval.nc \
    --backtest-params.n-splits 2 \
    --backtest-params.n-periods 3
```

Generate the evaluation plot:

```console
$ chap plot-backtest minimalist_eval.nc minimalist_evaluation_plot.png --plot-type evaluation_plot
```

**Evaluation plot** -- observed vs. predicted cases per location and split:

![Minimalist evaluation plot](assets/minimalist_evaluation_plot.png)

### 4. Evaluate Model B -- Auto-EWARS

Now evaluate a more sophisticated model. The EWARS model is fetched directly from GitHub:

```console
$ chap eval https://github.com/dhis2-chap/chap_auto_ewars \
    chap_LAO_admin1_monthly.csv \
    ewars_eval.nc \
    --backtest-params.n-splits 2 \
    --backtest-params.n-periods 3
```

Generate the evaluation plot:

```console
$ chap plot-backtest ewars_eval.nc ewars_evaluation_plot.png --plot-type evaluation_plot
```

**Evaluation plot:**

![EWARS evaluation plot](assets/ewars_evaluation_plot.png)

### 5. Compare models

Export aggregate metrics from both evaluations into a single CSV:

```console
$ chap export-metrics \
    --input-files minimalist_eval.nc \
    --input-files ewars_eval.nc \
    --output-file metrics_comparison.csv
```

The output CSV contains one row per model with columns for each metric:

| model | mae | rmse | crps | coverage_10_90 |
|---|---|---|---|---|
| minimalist_example_uv | 124.6 | 282.1 | 124.6 | 0.0 |
| chap_auto_ewars | 97.8 | 227.4 | 67.2 | 0.781 |

Lower MAE, RMSE, and CRPS indicate better accuracy. Coverage_10_90 measures how often the true value falls within the 10th-90th percentile prediction interval (ideal: 0.80).

In this comparison, the EWARS model outperforms the minimalist baseline across all metrics.
