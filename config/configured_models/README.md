# Configured Models Seeding

This directory contains YAML files that define which model templates and configured models are seeded into the database on startup.

## How it works

On startup, the REST API calls `seed_configured_models_from_config_dir()` (in `chap_core/database/model_template_seed.py`), which:

1. Parses `default.yaml` first, keeping only the **last version** listed for each model (earlier versions serve as historical documentation).
2. Parses all other `*.yaml` files in this directory (e.g. `local.yaml`, `benchmark_models.yaml`), keeping **all versions** listed.
3. For each model entry, fetches the model metadata (name, description, covariates, user options, etc.). How this happens depends on `uses_chapkit`:
   - **MLproject models** (`uses_chapkit: false`, the default): takes the last version, constructs a GitHub URL (`{url}@{commit}`) and fetches `MLProject.yaml` from the repository at that commit.
   - **Chapkit models** (`uses_chapkit: true`): treats `url` as the base URL of a running chapkit service, waits up to 30 seconds for it to report healthy, and reads the template config over HTTP. `versions` is ignored. If the service does not become healthy, the model is skipped with an error in the log and seeding continues.
4. Inserts a `ModelTemplateDB` row (or updates it if the name already exists).
5. For each configuration listed under `configurations:`, inserts a `ConfiguredModelDB` row with the specified user option values and additional covariates.
6. Finally, adds a built-in naive model template used for testing.

Models that already exist in the database (matched by name) are updated rather than duplicated, so seeding is idempotent.

## File format

```yaml
- url: https://github.com/org/model-repo
  versions:
    v1: "@<commit-sha>"          # historical, ignored in default.yaml
    v2: "@<commit-sha-or-branch>" # last entry is the one that gets seeded
  configurations:                 # optional, defaults to a single "default" config
    config_name:
      user_option_values:
        option_key: value
      additional_continuous_covariates:
        - rainfall
        - mean_temperature
```

### Fields

- **url**: For MLproject models, the GitHub repository URL. For chapkit models (`uses_chapkit: true`), the base URL of the running chapkit service, e.g. `http://my-model:8000` using the compose service name.
- **name** (optional): Overrides the template name declared by the model itself. Use it to avoid name clashes when seeding two variants of the same model.
- **uses_chapkit** (optional, default `false`): Set to `true` when `url` points at a running chapkit service rather than a GitHub repository. See [Running Your Own Model](../../docs/modeling-app/running-your-own-model.md).
- **versions**: Named versions mapping to git refs. Prefix with `@` for commits/branches. In `default.yaml`, only the last version is used; in other files, all versions are available. Ignored for chapkit models.
- **configurations** (optional): Named configurations for the model template. Each configuration can set `user_option_values` (model-specific parameters) and `additional_continuous_covariates`. If omitted, a single "default" configuration with empty values is created.

### Chapkit service example

Chapkit services normally do **not** belong in these files. They register themselves with Chap on startup via `SERVICEKIT_ORCHESTRATOR_URL`, which needs no entry here and no image rebuild -- see [Running Your Own Model](../../docs/modeling-app/running-your-own-model.md).

Seed a chapkit service from configuration only when it cannot be given the registration environment variables:

```yaml
# config/configured_models/local.yaml
- url: http://my-model:8000
  uses_chapkit: true
  versions:
    v1: "/v1"
  configurations:
    default:
      user_option_values: {}
```

The service must be reachable from inside the `chap` container and healthy when Chap starts, so declare it in the same Docker Compose project and give the `chap` service a `depends_on` on it (or accept that the first seeding attempt may miss it and restart Chap afterwards).

## Adding models

Do not edit `default.yaml` directly -- it is overwritten on updates. Instead, create a new `*.yaml` file (e.g. `local.yaml`) following the same format. Any `*.yaml` file in this directory (except files ending in `.disabled`) will be read on startup.

## Key source files

- `chap_core/models/local_configuration.py` -- YAML parsing logic
- `chap_core/database/model_template_seed.py` -- database seeding logic
- `chap_core/database/database.py` (`create_db_and_tables`) -- startup entry point that triggers seeding
