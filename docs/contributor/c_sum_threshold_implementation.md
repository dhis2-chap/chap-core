# WHO C-SUM threshold strategy — what changed

Branch: `feat/c-sum-threshold-strategy`  
Commit: `feat: add WHO C-SUM threshold strategy`  
Guide followed: [Creating Custom Threshold Strategies](creating_custom_threshold_strategies.md)

This note explains the Phase 1 chap-core work for teammates reviewing the PR.

## Goal

Close the WHO threshold gap in chap-core by adding a **`c_sum`** endemic-channel strategy. Existing strategies already cover:

| Strategy id | Role |
|-------------|------|
| `seasonal` | mean + k × std (same season) |
| `percentile` | WHO percentile / endemic channel band |
| **`c_sum` (new)** | WHO three-period C-SUM smoothed baseline |

Downstream CAPS can later call `POST /v1/analytics/thresholds` instead of computing C-SUM locally.

## What was added or changed

### New files

| File | Purpose |
|------|---------|
| `chap_core/assessment/thresholds/c_sum.py` | Strategy implementation + window math |
| `tests/evaluation/test_c_sum_thresholds.py` | Unit / golden tests (population SD vs CAPS) |

### Modified files

| File | Change |
|------|--------|
| `chap_core/assessment/thresholds/params.py` | `CSumParams`; extend `ThresholdParams` union |
| `chap_core/assessment/thresholds/__init__.py` | Import `c_sum` in `_discover_strategies()` |
| `tests/integration/rest_api/test_threshold_strategies.py` | Registry, shape, multi-line, baseline, validation for `c_sum` |
| `tests/integration/rest_api/test_from_seeded_engine.py` | OpenAPI discriminator mapping includes `c_sum` |

**Not changed:** `threshold_service.py` and the REST route handler logic — they pick up new strategies through the registry automatically.

## How the strategy works

```text
historical_observations [location, time_period, disease_cases]
        │
        ▼
filter_to_baseline(baseline_years)   ← same helper as percentile
        │
        ▼
For each (location, season):
  pool prev + current + next sub-period values across baseline years
        │
        ▼
line(s): mean + k × population SD, one per std_multiplier
        │
        ▼
align_seasonal_to_periods → [period_id, location, line, threshold]
```

### Semantics

- **Lines** — each line is mean + k × population SD of the pooled three-period window (not a running sum within one year). `std_multiplier` k = 0 gives the C-SUM mean, k = 1.96 the upper 95% limit.
- **Population SD** — divides by N, matching CAPS / WHO practice. The `seasonal` strategy uses the sample SD (N - 1).
- **Alarm levels** — which line counts as a warning or an alarm is chosen by the threshold policy, one line per level, not by this strategy.
- **Static channel** — like `percentile`: one line per `(location, season)` from the dataset baseline; requested `period_ids` only select which rows are returned.
- **Sparse windows** — fewer than 3 pooled values → row omitted; the API fills `null`.
- **Frequencies in v1** — month and week only (same as seasonal / percentile).

### Request shape

```json
{
  "dataset_id": 1,
  "period_ids": ["2023-01", "2023-06"],
  "params": {
    "type": "c_sum",
    "std_multiplier": [0, 1.96],
    "baseline_years": 5
  }
}
```

- `type` is required (no default) so OpenAPI clients can discriminate the union.
- `std_multiplier` defaults to `1.96`.
- `lines` in the response echoes the resolved multipliers in request order (e.g. `[0, 1.96]`).

### Discovery

After registration:

- `GET /v1/analytics/thresholds/strategies` lists `c_sum`.
- `POST /v1/analytics/thresholds` accepts `params.type = "c_sum"`.

## Plugin checklist (contributor guide)

1. `CSumParams` with `type: Literal["c_sum"]` (no default on `type`)
2. `lines` property (from `std_multiplier`)
3. `CSumThresholdStrategy` subclasses `ThresholdStrategyBase` and implements `compute`
4. `@threshold("c_sum", ...)` — id matches the `type` literal
5. Module imported in `_discover_strategies()`
6. Model added to `ThresholdParams` discriminated union
7. Tests cover registry union, shape, multi-line, validation; seasonal/percentile tests unchanged

## Testing done

| Command / suite | Result |
|-----------------|--------|
| `tests/evaluation/test_c_sum_thresholds.py` | Golden CAPS window `[10..60]` → mean 35, pop SD ≈ 17.078, upper ≈ 68.473 |
| `tests/integration/rest_api/test_threshold_strategies.py` | Includes `c_sum`; existing seasonal/percentile still pass |
| `test_builtin_strategies_are_in_params_union` | Green with `CSumParams` in the union |
| `test_threshold_params_schema_is_discriminated_union` | Mapping includes `c_sum` |
| `make check` | Ruff, mypy, pyright, alembic heads |
