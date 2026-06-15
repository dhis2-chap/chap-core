# Make your model configurable

In [Implement your own model from a minimalist example](../kigali-workshop/kigali-webinar-series/session-3/fork-example.md)
you forked a fixed model and changed its code. In this guide you take the next step:
building a model whose behaviour can be **configured from the outside**, without editing
the code each time.

By the end you will be able to:

- expose tunable options and extra covariates from a model,
- inspect a model's configuration spec with the CHAP CLI,
- run an evaluation with a specific configuration.

We build on the example model
[**chap-models/minimalist_configurable_model**](https://github.com/chap-models/minimalist_configurable_model).
Open its `README.md` first -- it explains how the pieces fit together. This guide is the
hands-on exercise that goes with it.

## Prerequisites

This tutorial is a continuation of the
[Modeling Webinar Series](../kigali-workshop/kigali-webinar-series/index.md) and the
[Kigali workshop material](../kigali-workshop/kigali-workshop-material/index.md). Before
starting, you should have worked through the following:

- **[Webinar Session 2](../kigali-workshop/kigali-webinar-series/session-2/index.md)** --
  you are comfortable on the [command line](../kigali-workshop/kigali-webinar-series/session-2/terminal.md),
  with [git and GitHub](../kigali-workshop/kigali-webinar-series/session-2/git-github.md)
  (forking and cloning), and with
  [virtual environments](../kigali-workshop/kigali-webinar-series/session-2/virtual-environments.md).
- **[Webinar Session 3](../kigali-workshop/kigali-webinar-series/session-3/index.md)** --
  you have [installed Chap](../kigali-workshop/kigali-webinar-series/session-3/install-chap.md)
  and have
  [run a minimalist example model](../kigali-workshop/kigali-webinar-series/session-3/fork-example.md)
  both in isolation and through Chap.
- **[Monday Afternoon workshop material](../kigali-workshop/kigali-workshop-material/monday-afternoon.md)** --
  you have run a backtest with `chap eval` and looked at the results.

It also helps (but is not required) to have gone through the
[Pre-session on working with datasets](../kigali-workshop/kigali-workshop-material/11_feb_presession.md),
so you understand the Chap-compatible CSV format (`time_period`, `location`,
`disease_cases`, plus covariate columns).

Concretely, you need:

- [uv](https://docs.astral.sh/uv/) installed.
- The Chap CLI installed and working (`chap --help` runs).
- A GitHub account, so you can fork the example model.

## What "configurable" means here

The example model fits a linear regression per location, predicting `disease_cases` from:

- **one required covariate** -- `rainfall`. CHAP refuses to run the model on a dataset
  that does not contain it.
- **any additional covariates** the caller chooses to attach, such as `mean_temperature`.
- **`n_lags`** -- a tunable parameter controlling how many lagged copies of each covariate
  are used as features.

All tunable options are declared once, as a Pydantic class in `config.py`:

```python
class ModelConfig(BaseModel):
    n_lags: int = Field(default=3, ge=0, le=12,
                        description="Number of lag periods added as features for each covariate.")
```

That class is the single source of truth. Its JSON schema is dumped into the
`user_options:` block of the `MLproject` file, which is what CHAP advertises to callers.
At runtime CHAP writes the chosen values to disk and the model validates them back into a
`ModelConfig`, logging exactly what it received.

## Fork and clone

Fork [chap-models/minimalist_configurable_model](https://github.com/chap-models/minimalist_configurable_model)
to your own account (see the GitHub guide from Session 2 if unsure), then clone it:

```console
git clone https://github.com/YOUR-USERNAME/minimalist_configurable_model.git
cd minimalist_configurable_model
```

Run it once in isolation to confirm your environment works. Note how the model logs the
configuration it received (here, the defaults):

```console
uv run python isolated_run.py
```

You should see lines like:

```
... | Received user options: {'n_lags': 3}
... | Additional continuous covariates: []
... | Using covariates: ['rainfall'] with n_lags=3
```

## Inspect the configuration spec

Before configuring a model, ask CHAP what options it exposes. The `chap model schema`
command reads the `user_options` from the `MLproject` and prints a JSON schema:

```console
chap model schema .
```

```yaml
properties:
  user_option_values:
    properties:
      n_lags:
        default: 3
        description: Number of lag periods added as features for each covariate.
        maximum: 12
        minimum: 0
        title: N Lags
        type: integer
    required: []
  additional_continuous_covariates:
    type: array
    items:
      type: string
    default: []
```

You can also generate a ready-to-edit example configuration filled with defaults:

```console
chap model schema . --example --output-file my_config.yaml
```

## Run an evaluation with a configuration

Edit `my_config.yaml` to choose a value for `n_lags` and to attach an additional
covariate:

```yaml
user_option_values:
  n_lags: 5
additional_continuous_covariates:
  - mean_temperature
```

Then run an evaluation, passing the configuration with `--model-configuration-yaml`:

```console
chap eval \
    --model-name . \
    --dataset-csv https://raw.githubusercontent.com/dhis2-chap/chap-core/master/example_data/laos_subset.csv \
    --output-file /tmp/eval.nc \
    --model-configuration-yaml my_config.yaml \
    --backtest-params.n-splits 2 \
    --backtest-params.n-periods 1
```

The model logs the `n_lags` and additional covariates it received, so you can confirm your
configuration reached it. Try a different `n_lags` and compare the results.

## Exercise: add a new configurable parameter

Your task is to add a second tunable option to the model: whether the linear regression
should fit an intercept.

1. **Add a field to `ModelConfig`** in `config.py`:

    ```python
    class ModelConfig(BaseModel):
        n_lags: int = Field(default=3, ge=0, le=12,
                            description="Number of lag periods added as features for each covariate.")
        fit_intercept: bool = Field(default=True,
                                    description="Whether the linear regression fits an intercept term.")
    ```

2. **Use it in `train.py`** where the model is created:

    ```python
    models[location] = LinearRegression(fit_intercept=config.fit_intercept).fit(features, target)
    ```

3. **Regenerate the schema and copy it into `MLproject`.** The `user_options:` block in
   `MLproject` is what CHAP advertises, but it is *not* updated automatically when you edit
   `ModelConfig` -- you have to copy it across yourself. First dump the schema from the
   Pydantic class:

    ```console
    uv run python dump_user_options.py
    ```

    This prints a `user_options:` block that now includes `fit_intercept`:

    ```yaml
    user_options:
      n_lags:
        default: 3
        description: Number of lag periods added as features for each covariate.
        maximum: 12
        minimum: 0
        title: N Lags
        type: integer
      fit_intercept:
        default: true
        description: Whether the linear regression fits an intercept term.
        title: Fit Intercept
        type: boolean
    ```

    Now **open `MLproject` and replace the existing `user_options:` block with this output**
    (the rest of the file -- `required_covariates`, `entry_points`, and so on -- stays
    unchanged). If you skip this copy step, `chap model schema` will keep showing the old
    options even though the code already understands the new one.

4. **Verify the new option appears** in the spec:

    ```console
    chap model schema .
    ```

    `fit_intercept` should now show up under `user_option_values`.

5. **Run an evaluation** with the new option set, e.g. add `fit_intercept: false` to
   `my_config.yaml` and re-run the `chap eval` command above.

6. **Commit and push** your changes to your fork.

### Verification

- `chap model schema .` lists both `n_lags` and `fit_intercept`.
- `chap eval ... --model-configuration-yaml my_config.yaml` completes and the model logs the
  configuration it received.
- Your changes are pushed to your GitHub fork.

## Get help

[Community of Practice for the webinar series](https://community.dhis2.org/c/development/chap/84)
