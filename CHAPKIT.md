# Chapkit Integration Map

Reference document for how chapkit integrates with chap-core across CLI, REST API, and database layers.

## Overview

**chapkit** is a standalone ML service framework that exposes train/predict operations over a REST API.
**chap-core** consumes chapkit services, wrapping them as model templates that plug into the existing evaluation, backtesting, and forecasting pipelines.

## Integration Modes

| Mode | Input | Behavior |
|------|-------|----------|
| **URL mode** | `http://host:port` | Connects to an already-running chapkit service |
| **Directory mode** | Local path | Auto-starts a chapkit dev server via `uv run fastapi dev`, manages lifecycle |

Detection: `is_url()` in `chapkit_service_manager.py` checks if the input looks like a URL. For URL mode, `_is_chapkit_url()` in `utils.py` probes `/api/v1/info` and validates against `MLServiceInfo` to auto-detect chapkit services without requiring `--run-config.is-chapkit-model`.

## File Map

### Model Layer

| File | Purpose |
|------|---------|
| `chap_core/models/external_chapkit_model.py` | `ExternalChapkitModelTemplate` (factory) + `ExternalChapkitModel` (train/predict calls) |
| `chap_core/models/chapkit_rest_api_wrapper.py` | `CHAPKitRestAPIWrapper` - synchronous HTTP client for all chapkit endpoints. `info()` returns typed `MLServiceInfo` |
| `chap_core/models/chapkit_service_manager.py` | `ChapkitServiceManager` - subprocess lifecycle, port allocation, health polling |
| `chap_core/models/utils.py` | `ModelTemplateType = ModelTemplate \| ExternalChapkitModelTemplate`; factory branch on `is_chapkit_model` or auto-detection via `_is_chapkit_url()` |
| `chap_core/models/model_template.py` | `from_directory_or_github_url()` accepts `is_chapkit_model` flag |

### CLI

| File | Purpose |
|------|---------|
| `chap_core/api_types.py` | `RunConfig.is_chapkit_model: bool` |
| `chap_core/cli_endpoints/_common.py` | `get_model()` passes flag to `ModelTemplate.from_directory_or_github_url()` |
| `chap_core/cli_endpoints/evaluate.py` | `chap eval` wires `run_config.is_chapkit_model` through to model loading |
| `chap_core/cli_endpoints/preference_learn.py` | Same pattern for preference learning commands |

### Database

| File | Purpose |
|------|---------|
| `chap_core/database/model_templates_and_config_tables.py` | `ConfiguredModelDB.uses_chapkit: bool` column |
| `chap_core/database/database.py` | `add_configured_model()` stores flag; `_get_model()` branches on it to load `ExternalChapkitModelTemplate` |
| `chap_core/database/model_template_seed.py` | Seeds chapkit models from YAML config, fetches metadata from running service |

### REST API

| File | Purpose |
|------|---------|
| `chap_core/rest_api/v2/routers/services.py` | Service registry endpoints: `$register`, `$ping`, list, get, delete |
| `chap_core/rest_api/v2/dependencies.py` | `Orchestrator` factory (Redis db=3), `X-Service-Key` auth |
| `chap_core/rest_api/services/orchestrator.py` | `Orchestrator` - Redis-backed registry with TTL-based expiration |
| `chap_core/rest_api/services/schemas.py` | `ServiceInfo`, `MLServiceInfo`, `RegistrationRequest/Response`, etc. |

### Configuration

| File | Purpose |
|------|---------|
| `chap_core/models/local_configuration.py` | `LocalModelTemplateWithConfigurations.uses_chapkit` for YAML parsing |
| `config/configured_models/default.yaml` | Model seed config (chapkit entries currently commented out) |

### Other

| File | Purpose |
|------|---------|
| `chap_core/exceptions.py` | `ChapkitServiceStartupError` |
| `tests/external/test_external_chapkit_model.py` | Unit tests for service manager and model template |
| `tests/integration/rest_api/test_chapkit_self_registration.py` | End-to-end tests for self-registration flow (fakeredis + in-memory SQLite) |
| `docs/external_models/chapkit.md` | User-facing chapkit integration guide |

## CLI Integration

### Tested Commands

Evaluate a chapkit model running at a URL (auto-detected, no flag needed):
```bash
chap eval --model-name http://127.0.0.1:8000 \
    --dataset-csv example_data/vietnam_monthly.csv \
    --output-file /tmp/chapkit_eval_test.nc
```

Evaluate with explicit flag (still supported as override):
```bash
chap eval --model-name http://127.0.0.1:8000 \
    --run-config.is-chapkit-model \
    --dataset-csv example_data/vietnam_monthly.csv \
    --output-file /tmp/chapkit_eval_test.nc
```

Evaluate a chapkit model from a local directory (auto-starts service, flag required):
```bash
chap eval --model-name /path/to/chapkit/model \
    --run-config.is-chapkit-model \
    --dataset-csv example_data/vietnam_monthly.csv \
    --output-file /tmp/chapkit_eval_test.nc
```

### Call Chain

1. `RunConfig.is_chapkit_model` parsed from CLI args (`api_types.py`)
2. `evaluate()` passes to `get_model()` (`_common.py`)
3. `get_model()` calls `ModelTemplate.from_directory_or_github_url(is_chapkit_model=...)`
4. Factory in `utils.py` checks: explicit flag OR (is URL AND `_is_chapkit_url()` auto-detection)
5. Creates `ExternalChapkitModelTemplate` instead of `ModelTemplate`
6. For directory mode, context manager starts/stops the subprocess

## REST API Integration

### v1: Backtest System

No chapkit-specific router. The `uses_chapkit` flag on `ConfiguredModelDB` controls model loading at runtime:

```
database.py:_get_model()
  -> if configured_model.uses_chapkit:
       ExternalChapkitModelTemplate(source_url).get_model(configured_model)
```

Backtest jobs use whatever model the configured model points to -- chapkit models are transparent once loaded.

### v2: Service Registry

Chapkit services self-register via the v2 API:

| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/v2/services/$register` | POST | `X-Service-Key` | Register with TTL (default 30s) |
| `/v2/services/{id}/$ping` | PUT | `X-Service-Key` | Keepalive, resets TTL |
| `/v2/services` | GET | None | List all live services |
| `/v2/services/{id}` | GET | None | Get service details |
| `/v2/services/{id}` | DELETE | `X-Service-Key` | Deregister |

Backend: Redis db=3, key prefix per service, automatic TTL expiration.
Auth: Optional `SERVICEKIT_REGISTRATION_KEY` env var; if unset, auth is skipped.

## Database

`ConfiguredModelDB.uses_chapkit: bool` (default `False`) on the configured models table.

Seeding flow (`model_template_seed.py`):
1. Parse YAML config with `uses_chapkit: true`
2. Create `ExternalChapkitModelTemplate` from URL
3. Wait for healthy (30s timeout)
4. Fetch model metadata via `/api/v1/info` and `/api/v1/configs/$schema`
5. Store template + configured models with `uses_chapkit=True`

Can be skipped with `skip_chapkit_models=True` flag during seeding.

## Data Flow: Train/Predict

```
ExternalChapkitModelTemplate
  |
  |-- get_model(configured_model) -> ExternalChapkitModel
  |     |
  |     |-- .train(train_data)
  |     |     POST /api/v1/ml/$train -> job_id
  |     |     poll_job(job_id) until complete
  |     |     return trained artifacts
  |     |
  |     |-- .predict(predict_data)
  |           POST /api/v1/ml/$predict -> job_id
  |           poll_job(job_id) until complete
  |           get_prediction_artifact_dataframe() -> DataFrame
```

All HTTP calls go through `CHAPKitRestAPIWrapper`. Jobs are async on the chapkit side; the wrapper polls until completion.

## Known Issues

- **Workspace snowball bug (chapkit)**: `FunctionalModelRunner.on_train()` and `ShellModelRunner.on_train()` call `prepare_workspace(Path.cwd(), workspace_dir)` which copies the entire project directory, then zips it into a blob stored in SQLite via `PickleType`. If the chapkit service's SQLite DB file (e.g. `data/chapkit.db`) lives inside the project directory, each run the DB contains previous artifact blobs, so the workspace zip grows exponentially. First eval succeeds, second fails with `sqlite3.DataError: string or blob too big`. Confirmed with `hello-world` model where `data/chapkit.db` is inside the project root. Fix needed in chapkit: either add `"*.db"` / `"data"` to `WORKSPACE_EXCLUDE_PATTERNS` in `runner.py`, or store the DB outside the project directory.
- **Period type mismatch**: Chapkit uses "monthly"/"weekly" while chap-core uses "month"/"week". The mapping is handled by `_chapkit_period_to_chap()` in `external_chapkit_model.py`.

## Conceptual Mapping

| chapkit | chap-core | Notes |
|---------|-----------|-------|
| service | ModelTemplateDB | A chapkit service = a model template |
| service + config | ConfiguredModelDB | A configured model = template + user config |
| train/predict calls | BackTest | Runs stored as BackTest in DB. Could store chapkit artifact_id reference later |

## Roadmap: Self-Registration

- [x] Use chapkit `id` field as model template name (instead of `{display_name}_v{version}`)
- [x] Map `version` and `repository_url` from chapkit metadata to model template config
- [x] Extract `ml_service_info_to_model_template_config()` as standalone converter (works with both chapkit and local MLServiceInfo)
- [x] Sync live chapkit services to DB on GET /model-templates via `_sync_live_chapkit_services()` in crud.py (uses v2 Orchestrator, graceful fallback if Redis unavailable)
- [x] Auto-sync chapkit configs to configured models on first discovery via `_sync_chapkit_configured_models()`
- [x] Health status on model templates: `health_status` field in GET /model-templates response, set to `"live"` for services in v2 registry (no extra HTTP calls)
- [x] Map `requires_geo` from chapkit to chap-core (DB column + Alembic migration `b2c3d4e5f6a7`)
- [x] Map `documentation_url` from chapkit metadata (DB column + Alembic migration `b2c3d4e5f6a7`)

### Testing Self-Registration

Start a chapkit service and register it with the v2 service registry, then verify it syncs to model templates:

```bash
# Start a chapkit service (e.g. hello-world model)
cd /path/to/hello-world && uv run fastapi dev

# Register with chap-core's v2 service registry (POST to v2 endpoint)
curl -X POST http://127.0.0.1:8001/v2/services/\$register \
  -H "Content-Type: application/json" \
  -d '{"url": "http://127.0.0.1:8000", "info": <MLServiceInfo from /api/v1/info>}'

# Verify it shows up in model templates (triggers sync from registry to DB)
curl http://127.0.0.1:8001/v1/crud/model-templates | jq '.[].name'
```

#### Automated Tests

Integration tests cover the full self-registration flow (register -> sync -> verify model templates and configured models) using fakeredis and in-memory SQLite:

```bash
pytest tests/integration/rest_api/test_chapkit_self_registration.py -v
```

## Cleanup Opportunities

- **`is_chapkit_model` flag threading**: The flag is still passed through 4-5 layers (CLI args -> RunConfig -> get_model -> ModelTemplate -> utils). Auto-detection now handles URL mode, but the flag is still needed for directory mode. Could be further simplified.
- **Commented-out config**: `default.yaml` has commented chapkit entries -- decide whether to remove or document as examples.
- **Deprecated evaluate()**: `cli_endpoints/evaluate.py` has a deprecated `evaluate()` function that still carries `is_chapkit_model` -- can be removed when the old codepath is dropped.
- **v1/v2 gap**: Resolved for model template sync, configured model sync, and health status. Remaining: backtest runs don't yet store chapkit `artifact_id` references.
