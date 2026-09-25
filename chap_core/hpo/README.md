# Hyperparameter Optimization (`hpo`)

The `hpo` package implements hyperparameter optimization for CHAP models.

It is responsible for:

* defining and parsing hyperparameter search spaces,
* proposing candidate configurations,
* evaluating candidates,
* coordinating the optimization loop,
* selecting the best configuration, and
* returning HPO results and trial metadata.

HPO is not tied to the CLI. The CLI is one integration point, while the optimization itself can also be constructed and used programmatically.

The CLI evaluation command, `Evaluation`, and `get_hpo_estimator(...)` live outside the `hpo` package.

## Main components

### `HyperparameterOptimizer`

`HyperparameterOptimizer` coordinates the optimization run.

It:

1. resets the configured `Searcher`,
2. asks for candidate hyperparameters,
3. combines each candidate with the optional base `ModelConfiguration`,
4. evaluates the candidate using `Objective`,
5. reports the result back to the searcher,
6. records the trial,
7. stops when the trial budget is reached or the search space is exhausted,
8. selects the best successful trial.

The optimizer implements `MetaLearner`, meaning that it receives a dataset and produces a tuned model configuration rather than directly producing forecasts.

```python
result = optimizer.meta_learn(dataset)
```

The returned `HyperparameterOptimization` contains the best configuration, best score, trial leaderboard, runtime information, search settings, and stop reason.

### `Searcher`

A `Searcher` is responsible for proposing hyperparameter configurations.

The interface follows an `ask` / `tell` pattern:

```python
searcher.reset(search_space, seed)

candidate = searcher.ask()
searcher.tell(candidate, score)
```

Current implementations are:

* `GridSearcher`
* `RandomSearcher`
* `TPESearcher`

#### Grid search

Enumerates the Cartesian product of an explicit list-based search space.

```yaml
max_depth:
  values: [3, 5, 7]

learning_rate:
  values: [0.001, 0.01]
```

Grid search ends naturally when all combinations have been evaluated.

#### Random search

Samples independently from categorical, integer, and floating-point search-space specifications.

Random search samples with replacement and therefore requires a finite `max_trials`.

A supplied seed makes candidate sampling reproducible.

#### TPE search

Uses Optuna's Tree-structured Parzen Estimator sampler.

TPE is adaptive: scores from completed trials are passed back to Optuna and influence later candidates.

Failed trials are reported to Optuna as failed trials.

TPE also requires a finite `max_trials`.

### `Objective`

`Objective` maps a concrete model configuration to a scalar optimization score:

```text
ModelConfiguration + DataSet -> float
```

For each candidate it:

1. creates the configured model,
2. instantiates the estimator,
3. runs a validation backtest using `Evaluation.create(...)`,
4. calculates the selected metric,
5. returns the metric value.

The default metric is `rmse`.

The optimization direction is derived from the metric, allowing metrics that should either be minimized or maximized.

`Evaluation` itself belongs to the assessment layer, not to `hpo`; `Objective` reuses it to evaluate candidate configurations.

### Search space and result types

The package also contains the internal representations used by the components above.

Search spaces support:

* categorical values,
* integer ranges,
* floating-point ranges,
* stepped numeric ranges,
* logarithmic numeric ranges.

For example:

```yaml
learning_rate:
  type: float
  low: 0.0001
  high: 0.1
  log: true

max_depth:
  type: int
  low: 2
  high: 10

solver:
  values:
    - adam
    - sgd
```

The parsed representation is approximately:

```python
{
    "learning_rate": Float(
        low=0.0001,
        high=0.1,
        log=True,
    ),
    "max_depth": Int(
        low=2,
        high=10,
    ),
    "solver": ["adam", "sgd"],
}
```

The HPO result contains:

```text
best configuration
best parameters
best score
leaderboard
search settings
runtime
stop reason
```

Each trial records its configuration, score, runtime, and optional failure.

## Call flow

For CLI-based evaluation, the current flow is:

```text
chap eval
    |
    v
CLI evaluation command
    |                                           outside hpo/
    | estimator mode == HPO
    v
get_hpo_estimator(...)
    |
    |-- load search space
    |-- create Objective
    |-- create Searcher
    |-- create HyperparameterOptimizer
    v
Evaluation.create(...)
    |                                           outside hpo/
    |-- create outer train/test split
    |
    |-- HyperparameterOptimizer.meta_learn(train_set)
    |       |
    |       |-- Searcher.reset(...)
    |       |
    |       +--> Searcher.ask()
    |       |       |
    |       |       v
    |       |   candidate parameters
    |       |       |
    |       |       v
    |       |   build ModelConfiguration
    |       |       |
    |       |       v
    |       |   Objective(configuration, train_set)
    |       |       |
    |       |       |-- instantiate model
    |       |       |-- Evaluation.create(...)
    |       |       |-- calculate metric
    |       |       v
    |       |      score
    |       |       |
    |       |       v
    |       |   Searcher.tell(candidate, score)
    |       |
    |       +--> repeat until stop condition
    |       |
    |       v
    |   HyperparameterOptimization
    |
    |-- instantiate model using best configuration
    |-- run normal outer backtest
    |-- attach HPO metadata to evaluation
    v
Evaluation
```

The important distinction is between the two evaluation levels:

```text
outer evaluation
    final evaluation of the tuned model

inner validation
    evaluation performed by Objective for each HPO candidate
```

HPO runs on the outer training data. The final outer test data is therefore not directly used to select hyperparameters.

## CLI usage

HPO can be enabled through the normal `chap eval` command by setting the estimator mode to `hpo`.

For example:

```bash
chap eval \
  --model-name https://github.com/chap-models/minimal_template_example \
  --dataset-csv ./example_data/vietnam_monthly.csv \
  --output-file ./results/eval.nc \
  --estimator-options.mode hpo \
  --estimator-options.search-space ./search_space.yaml \
  --estimator-options.metric rmse \
  --estimator-options.searcher tpe \
  --estimator-options.max-trials 50 \
  --estimator-options.seed 42
```

The main HPO-related options are:

```text
--estimator-options.mode
    Select HPO mode.

--estimator-options.search-space
    Optional path to a YAML search space.
    If omitted, the model template's built-in hpo_search_space is used.

--estimator-options.metric
    Metric optimized by the objective.

--estimator-options.searcher
    Search strategy, for example grid, random, or tpe.

--estimator-options.max-trials
    Maximum number of trials.

--estimator-options.seed
    Optional random seed.
```

A simple search-space file could look like:

```yaml
learning_rate:
  type: float
  low: 0.0001
  high: 0.1
  log: true

max_depth:
  type: int
  low: 2
  high: 10
```

The CLI itself does not implement HPO. It loads the required inputs and delegates construction to `get_hpo_estimator(...)`, after which the normal evaluation workflow runs the optimizer.

## Configuration handling

A candidate contains only the hyperparameters selected by the searcher.

If no base model configuration was supplied, the candidate becomes the model's `user_option_values`.

If a base configuration exists, candidate parameters are overlaid onto it:

```python
config.user_option_values = {
    **(config.user_option_values or {}),
    **params,
}
```

This preserves configuration that is not part of the search space, while HPO values override matching user options.

## Trial failures

Failure of one candidate does not terminate the complete optimization.

A failed trial is recorded with:

```python
{
    "score": None,
    "failure": "<ExceptionType>: <message>",
}
```

The failure is also reported to the searcher.

Successful trials are ranked according to the optimization direction. Failed trials are placed after successful trials.

If every trial fails, the optimizer raises an error.

## Stop conditions

An HPO run stops for one of two reasons:

```text
max_trials
search_exhausted
```

`max_trials` is normally used by random search and TPE.

`search_exhausted` is normally produced by grid search after every combination has been evaluated.

## Integration boundary

The package boundary is intentionally:

```text
                 outside hpo/
                       |
        +--------------+--------------+
        |                             |
       CLI                    Evaluation / factory
        |                             |
        +--------------+--------------+
                       |
                       v
              HyperparameterOptimizer
                  /            \
                 v              v
             Searcher       Objective
                                |
                                v
                         Evaluation.create(...)
                           outside hpo/
```

The `hpo` package owns optimization behavior.

The surrounding application owns:

* CLI arguments,
* selecting HPO mode,
* constructing the optimizer,
* dataset loading,
* the outer evaluation,
* backtesting infrastructure,
* metric implementation,
* persistence of evaluation output.

This keeps HPO reusable independently of any particular entry point.
