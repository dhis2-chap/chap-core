# Design: Per-Covariate Lag Adjustment in EWARS Models

**Jira**: [CLIM-562](https://dhis2.atlassian.net/browse/CLIM-562)
**Status**: Draft
**Author**: Halvard Emil Sand-Larsen (assigned), design by CHAP team

## 1. Problem Statement

EWARS models currently apply a single `n_lags` parameter uniformly to all covariates. When `n_lags=3`, both rainfall and mean temperature get the same 3-period lag structure in the distributed lag non-linear model (DLNM) cross-basis.

This is a limitation because different climate variables influence disease incidence on different timescales. For example:

- **Rainfall** may affect mosquito breeding sites with a longer delay (e.g., 4-6 weeks), as standing water accumulates and larvae develop.
- **Temperature** may have a more immediate effect on vector activity and pathogen development (e.g., 1-3 weeks).

Allowing per-covariate lag adjustment would let the model capture these distinct temporal dynamics and improve predictive performance.

## 2. Current Architecture

### 2.1 Configuration Flow

The configuration passes through chap-core as follows:

```
User/HPO -> ConfiguredModelDB.user_option_values (JSON dict in DB)
         -> model_configuration_for_run.yaml (written to model working dir)
         -> External model reads YAML and applies parameters
```

Key: chap-core treats `user_option_values` as an **opaque JSON blob**. It validates the blob against the `user_options` JSON Schema from the model template, then passes it through unchanged. chap-core does not interpret the contents.

### 2.2 Where `n_lags` is Defined

**Model template** (`config/configured_models/default.yaml`, lines 46-53):
```yaml
- url: https://github.com/dhis2-chap/ewars_template
  configurations:
    default:
      user_option_values:
        n_lags: 3
        precision: 1
      additional_continuous_covariates:
        - rainfall
        - mean_temperature
```

**HPO search space** (`example_data/preference_learning/ewars_hpo_search_space.yaml`):
```yaml
n_lags:
  low: 1
  high: 6
  type: int
```

### 2.3 Where `n_lags` is Consumed (R Model Code)

In the EWARS R implementation (`lib.R`), the cross-basis is constructed with hardcoded lag values:

```r
get_crossbasis <- function(var, group, nlag){
    tsModel::Lag(var, group = group, k = 0:nlag)
    lagknot = equalknots(0:nlag, 2)
    basis <- crossbasis(var,
        argvar = list(fun = "ns", knots = equalknots(var, 2)),
        arglag = list(fun = "ns", knots = nlag/2))
}

# Called with hardcoded nlag=3 for both covariates:
extra_fields <- function(df) {
    basis_meantemperature <- get_crossbasis(df$meantemperature, df$ID_spat, 3)
    # ...
}
get_basis_rainfall <- function(df) {
    basis <- get_crossbasis(df$rainfall, df$ID_spat, 3)
    # ...
}
```

The `nlag` parameter in `get_crossbasis` controls how many lag periods are included in the DLNM cross-basis function. The cross-basis uses natural spline basis functions for both the variable dimension (`argvar`) and the lag dimension (`arglag`), with knots placed at `nlag/2`.

### 2.4 Config Validation in chap-core

Validation happens in `chap_core/database/model_templates_and_config_tables.py` (lines 77-87):

```text
@classmethod
def _validate_model_configuration(cls, user_options, user_option_values):
    schema = {
        "type": "object",
        "properties": user_options,
        "required": [...fields without defaults...],
        "additionalProperties": False,
    }
    jsonschema.validate(instance=user_option_values, schema=schema)
```

This uses `jsonschema.validate()` which natively supports `oneOf`, `anyOf`, and nested object schemas. No changes needed for the MVP.

### 2.5 HPO Search Space Loading

In `chap_core/hpo/base.py` (lines 61-103), `load_search_space_from_config()` parses the search space YAML into `Int` or `Float` dataclasses. It expects flat scalar parameters only:

```text
def load_search_space_from_config(config: dict) -> dict[str, Any]:
    space: dict[str, Any] = {}
    for name, spec in config.items():
        # ... parses into Int(low, high, step, log) or Float(...)
    return space
```

The HPO searchers (`GridSearcher`, `RandomSearcher`, `TPESearcher`) all work with this flat `dict[str, Int|Float|list]` and return flat `dict[str, scalar]` params. They have no concept of structured/nested parameters.

### 2.6 Best Config Assembly

In `chap_core/hpo/hpoModel.py` (line 108), best params are wrapped directly:

```text
self._best_config = {"user_option_values": best_params}
```

Where `best_params` is the flat dict from the searcher, e.g., `{"n_lags": 3, "precision": 0.5}`.

## 3. MVP: Per-Covariate Lags in EWARS Models

**Scope**: Changes only in external EWARS model repos. Zero chap-core changes.
**Estimated effort**: ~2 days

### 3.1 Config Schema Design

The model's config schema changes `n_lags` to accept both formats using JSON Schema `oneOf`:

```json
{
  "n_lags": {
    "title": "Number of lags",
    "oneOf": [
      {"type": "integer", "minimum": 1, "description": "Single lag applied to all covariates"},
      {
        "type": "object",
        "additionalProperties": {"type": "integer", "minimum": 1},
        "description": "Per-covariate lag values, e.g. {rainfall: 4, mean_temperature: 2}"
      }
    ],
    "default": 3
  }
}
```

**Backwards compatibility**: Existing configs with `n_lags: 3` (integer) continue to work unchanged.

### 3.2 New Config Format

```yaml
# Legacy (still works):
user_option_values:
  n_lags: 3
  precision: 1

# New per-covariate format:
user_option_values:
  n_lags:
    rainfall: 4
    mean_temperature: 2
  precision: 1
```

### 3.3 Model Code Changes

The model code needs a small helper to resolve the lag for a given covariate:

**R pseudocode** (for `chap_ewars_weekly`, `chap_auto_ewars`):
```r
resolve_nlag <- function(nlag_config, var_name, default = 3) {
  if (is.numeric(nlag_config)) return(nlag_config)
  if (is.list(nlag_config) && var_name %in% names(nlag_config)) {
    return(nlag_config[[var_name]])
  }
  return(default)
}

# Usage:
nlag_rain <- resolve_nlag(config$n_lags, "rainfall")
nlag_temp <- resolve_nlag(config$n_lags, "mean_temperature")
basis_rain <- get_crossbasis(df$rainfall, df$ID_spat, nlag_rain)
basis_temp <- get_crossbasis(df$meantemperature, df$ID_spat, nlag_temp)
```

**Python** (for `ewars_template` chapkit service): Same logic -- if `n_lags` is an int, apply to all; if it's a dict, look up per covariate.

### 3.4 Repos Affected

- `dhis2-chap/ewars_template` (chapkit Python service)
- `dhis2-chap/chap_auto_ewars` (R/INLA monthly)
- `dhis2-chap/chap_auto_ewars_weekly` (R/INLA weekly)

### 3.5 Why No chap-core Changes

1. **Config serialization**: `user_option_values` is a JSON column. A nested dict like `{"n_lags": {"rainfall": 4}}` serializes fine.
2. **Validation**: `jsonschema.validate()` handles `oneOf` schemas natively.
3. **Config passing**: The dict is written to YAML as-is via `model_configuration_for_run.yaml`. Nested dicts serialize correctly.
4. **Schema parsing**: `_parse_user_options_from_config_schema()` in `external_chapkit_model.py` returns the `user_options` dict as-is -- it doesn't interpret individual field schemas.

## 4. Full Solution: chap-core + Modeling App

**Scope**: General support for per-covariate parameters across the platform.
**Estimated effort**: ~5-8 days (needs further discussion)

### 4.1 HPO Per-Covariate Search

The main gap is HPO. The searchers only handle flat scalar params. To search per-covariate lag values, we need expand/collapse logic at the HPO boundary.

**Proposed search space format**:
```yaml
n_lags:
  per_covariate: true
  covariates: [rainfall, mean_temperature]
  low: 1
  high: 6
  type: int
```

**Expansion** (in `load_search_space_from_config()` in `chap_core/hpo/base.py`):
- When `per_covariate: true` is detected, expand into flat keys:
  - `n_lags__rainfall: Int(low=1, high=6)`
  - `n_lags__mean_temperature: Int(low=1, high=6)`
- Searchers work on these flat keys as usual (no searcher changes).

**Collapse** (in `get_leaderboard()` in `chap_core/hpo/hpoModel.py`):
- After HPO finds best params like `{"n_lags__rainfall": 4, "n_lags__mean_temperature": 2, "precision": 0.01}`:
- Collapse double-underscore keys back: `{"n_lags": {"rainfall": 4, "mean_temperature": 2}, "precision": 0.01}`
- This collapsed dict is stored in `user_option_values`.

**Files to modify**:
- `chap_core/hpo/base.py` -- add expansion in `load_search_space_from_config()`
- `chap_core/hpo/hpoModel.py` -- add collapse in `get_leaderboard()` (around line 108)
- `chap_core/hpo/objective.py` -- ensure collapsed config reaches model correctly

**Alternative**: The simpler alternative is explicit flat keys in the search space:
```yaml
n_lags__rainfall:
  low: 1
  high: 6
  type: int
n_lags__mean_temperature:
  low: 1
  high: 4
  type: int
```
This requires only the collapse step (no expansion), but the user must manually list each covariate.

### 4.2 Modeling App UI

The Modeling App (separate DHIS2 app repo) reads `user_options` from the chap-core REST API and renders form fields for each parameter.

For per-covariate params, the UI would need to:
1. Detect `oneOf` schemas where one option is `type: object`
2. Render a toggle between "single value for all" and "per-covariate" mode
3. In per-covariate mode, render one numeric input per covariate (using `required_covariates` from the model template to know which covariates exist)

This is a separate repo and a separate effort.

### 4.3 Optional: Covariate Name Validation

An optional enhancement in chap-core: when `n_lags` is a dict, validate that the keys match the model's `required_covariates` list. This would catch typos like `{"rainfal": 4}` early.

This could be added in `_validate_model_configuration()` in `model_templates_and_config_tables.py`, but requires the validation method to have access to the model template's `required_covariates`, which it currently does not.

## 5. Effort Estimates

| Approach | Scope | Repos | Effort |
|----------|-------|-------|--------|
| MVP | Per-covariate lags in EWARS models | `ewars_template`, `chap_auto_ewars`, `chap_auto_ewars_weekly` | ~2 days |
| Full | HPO + validation + UI | chap-core, Modeling App | ~5-8 days |

**Recommendation**: Start with the MVP. It is self-contained, backwards compatible, and immediately useful. The full solution requires further design discussion (particularly around the HPO expand/collapse convention and the Modeling App UI) and can be tackled when more models need per-covariate parameters.

## 6. Open Questions

1. Should per-covariate lag ranges be independently configurable in HPO (e.g., rainfall 1-6, temperature 1-3), or should they share the same range?
2. Should the full solution be generalized beyond `n_lags`? Other parameters might also benefit from per-covariate values (e.g., per-covariate smoothing, basis function types).
3. For the Modeling App UI, should we support arbitrary per-covariate params or only `n_lags` specifically?
