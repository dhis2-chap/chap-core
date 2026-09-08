# Pluggable Future-Weather Providers

Tracking issue: CLIM-1042

## Background

CHAP models that use climate covariates need weather data for the periods they
are forecasting into - data that does not exist yet at prediction time. Today
CHAP handles this inconsistently, and the way it does it during evaluation
differs from the way it does it in production.

Concretely, there are three different behaviours in the code right now:

- **Backtesting (default path):** when no provider is supplied, the splitting
  code hands the model the *actual observed* weather for the forecast window
  (`chap_core/assessment/dataset_splitting.py:177`). This is a look-ahead leak:
  the backtest scores a model that was given perfect weather knowledge it will
  never have in production.
- **Prediction via the REST API:** hardcoded to `QuickForecastFetcher`
  (`chap_core/rest_api/db_worker_functions.py:124`), a simple seasonal
  climatology model fitted on the historical data.
- **A `FutureWeatherFetcher` base class exists**
  (`chap_core/climate_predictor.py:69`) with a few implementations, but it is
  not a real abstraction: it is not configurable, not selectable by name, one
  implementation (`SeasonalForecastFetcher`) is an unimplemented stub, and the
  API/CLI request models have no field for choosing one.

The consequence is that backtest numbers are not comparable to real-world
performance, and users cannot bring their own forecasts.

## What we want

### 1. A future-weather provider plugin system

Future-weather providers should be plugins in exactly the same sense as metrics,
backtest plots and threshold strategies: a class with an `id`, a `name` and a
`description`, registered in a global registry by a decorator, discovered on
import, and selectable by id from the CLI and the REST API. The endpoint and
evaluation code should never need editing to add a provider.

The closest existing model is the threshold strategy registry
(`chap_core/assessment/thresholds/`), which is small and recent:

```python
class FutureWeatherProviderBase(ABC):
    id: str = ""
    name: str = ""
    description: str = ""
    leaks_future_data: bool = False

    @abstractmethod
    def get_future_weather(
        self,
        historical_data: DataSet,
        period_range: PeriodRange,
        future_data: DataSet | None = None,
        params: dict | None = None,
    ) -> DataSet[SimpleClimateData]: ...


@weather_provider(
    "climatology",
    "Seasonal climatology",
    "Per-location seasonal means fitted on the dataset's own history.",
)
class ClimatologyWeatherProvider(FutureWeatherProviderBase):
    def get_future_weather(self, historical_data, period_range, future_data=None, params=None):
        ...
```

with the usual registry accessors alongside it, mirroring
`get_threshold_strategy` / `list_threshold_strategies`:

- `get_weather_provider(provider_id)`
- `get_weather_providers_registry()`
- `list_weather_providers()` returning `{id, name, description, leaks_future_data}`
- `_discover_providers()` importing the built-in provider modules

### The `leaks_future_data` flag

Every provider declares whether it needs to see data from the forecast window
itself. This is the mechanism that makes look-ahead explicit instead of
accidental:

- `leaks_future_data = False` (the default) - the provider is given only the
  historical data and the period range it must fill. It is **structurally
  incapable** of leaking, because the future observations are never passed to
  it.
- `leaks_future_data = True` - the provider is additionally handed the actual
  observations for the forecast window in the `future_data` argument.

The call site decides what to pass based on the flag, so the guarantee is
enforced by the framework rather than by convention:

```python
provider_cls = get_weather_provider(provider_id)
future_weather = provider_cls().get_future_weather(
    historical_data,
    period_range,
    future_data=future_data if provider_cls.leaks_future_data else None,
    params=provider_params,
)
```

The `observed` provider is then an ordinary plugin with
`leaks_future_data = True` that returns `future_data` with the target column
removed - rather than today's special case where "use the real weather" is
encoded as the *absence* of a provider.

The flag should be carried through into `list_weather_providers()`, the REST
listing, and the stored backtest record, so that an evaluation run with a
leaking provider is visibly marked as diagnostic rather than comparable.

Providers we want to cover:

- **`climatology`** - statistical forecaster fitted on the dataset's own
  history. This is what `QuickForecastFetcher` does today.
- **Real seasonal forecast products** (e.g. ECMWF-style seasonal forecasts)
  pulled through CHAP's existing data-fetching layer.
- **`imported`** - forecasts supplied by the user or an external system as a
  dataset, so an implementing partner can evaluate against the forecast product
  they actually run on.
- **`observed`** - the actual observed weather ("perfect foresight"), the only
  built-in with `leaks_future_data = True`, kept as an explicit, opt-in choice
  for diagnostic runs rather than the silent default.

Note that this requires a small change to the current calling convention. Today
the provider is constructed with the historical data
(`future_weather_provider(hd).get_future_weather(fd.period_range)`), so the
class cannot be instantiated from a registry lookup alone. As with the threshold
strategies, the historical data should move from `__init__` into the method
signature so that the class is stateless and constructible by id. Provider-specific
configuration (which imported dataset, which seasonal product, which variables)
goes in the `params` dict, the same way threshold strategies take their
parameters.

### 2. A provider field on the request models

A field on `BacktestParams` and `PredictionParams` that names which provider to
use, validated against the registry, so the choice is recorded in the request,
in the run config, and in the stored backtest - not implied by which code path
was taken. Optional `params` alongside it for provider-specific configuration.

The available providers should be listable over the REST API, the way threshold
strategies are exposed at `GET /v1/analytics/thresholds/strategies`, so a client
can populate a picker without hardcoding ids.

### 3. The same provider in evaluation and in prediction

This is the core requirement. If a model is backtested with provider X,
predicting with that model must use provider X, and any mismatch should be
visible (recorded on the backtest/prediction record) rather than silent.

## Prior art

There is an unmerged branch, `feat/eval-future-covariate-source` (single commit
`9e09416d`, based on a June 2026 master), that solves a subset of this. It adds
`BacktestParams.future_covariate_source: Literal["real", "forecast"]`, a
`resolve_future_weather_provider()` helper, wires it through
`Evaluation.from_estimator`, the `chap eval` dry-run path and the REST backtest
worker, flips the default to `"forecast"`, and adds splitting tests that assert
observed vs. substituted climate.

It is worth reusing the wiring and the tests, but the design here differs in
three ways:

- The source is a closed `Literal`, not a registry, so adding a provider means
  editing `api_types.py` and `resolve_future_weather_provider()`. There is no
  id/name/description, and no way for a client to discover the options.
- `"real"` resolves to `None`, which falls through to the leaking `else` branch
  in `train_test_generator`. The leak stays implicit; the `leaks_future_data`
  flag above is what turns it into a declared property of a named provider.
- It covers only the backtest side. `PredictionParams` is untouched, so nothing
  ties the provider used at prediction time to the one used in evaluation -
  which is the core requirement here.

## Why this matters

- Backtest results today systematically overstate performance for
  climate-driven models, because evaluation leaks the true future weather.
- Partners who have their own operational forecast feeds cannot evaluate CHAP
  models under the conditions they will actually deploy under.
- Comparing two models is only meaningful if both saw the same kind of
  future-weather input; without a recorded provider we cannot assert that.

## Scope and deliverables

- `FutureWeatherProviderBase` plus a `@weather_provider(id, name, description)`
  decorator and registry, following the threshold strategy module layout.
- Built-in providers: `climatology` (existing behaviour), `imported`
  (user-supplied forecasts), `observed` (explicit opt-in). A real
  seasonal-forecast implementation may be split into a follow-up issue.
- Move historical data out of the provider constructor and into
  `get_future_weather`, so providers can be resolved by id.
- A `leaks_future_data` class flag, with the call site passing `future_data`
  only to providers that declare it, and the flag surfaced in the registry
  listing, the REST listing, and the stored backtest record.
- New provider field on `BacktestParams` and `PredictionParams`, threaded
  through the CLI (`chap eval`), the REST API, and the worker functions.
- A REST endpoint listing the registered providers.
- A contributor guide, alongside the existing "Creating Custom Metrics" and
  "Creating Custom Threshold Strategies" pages.
- Provider identity (and params) persisted with backtest and prediction results.
- Decision needed: what the **default** becomes. Changing the backtest default
  away from observed weather will change existing evaluation numbers - this
  needs a call and probably a migration note.

## Out of scope

- Ingesting any specific vendor's forecast product end-to-end.
- Changing how models themselves consume the weather covariates.

## Open questions

- Should the provider be settable per model template (some models may require a
  specific one), or purely per run?
- Do we hard-fail when a prediction's provider differs from the one used in the
  backtest, or warn?
- Do imported forecasts need to be versioned/stored as datasets in the DB, or
  passed by reference per run?
- Should provider `params` be free-form (as for threshold strategies) or typed
  per provider? Typed would let the API validate and let clients build a form,
  at the cost of more machinery.
